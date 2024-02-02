from logging import INFO, Formatter, Logger, StreamHandler, getLogger

import geopandas as gpd
import typer
from dask.distributed import Client
from dep_tools.azure import blob_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.loaders import OdcLoaderMixin, StackXrLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.stac_utils import set_stac_properties

# from dep_tools.task import SimpleLoggingAreaTask
from dep_tools.task import SimpleLoggingAreaTask
from dep_tools.utils import search_across_180
from dep_tools.writers import AzureDsWriter
from typing_extensions import Annotated
from xarray import DataArray, Dataset, merge


def get_logger(region_code: str) -> Logger:
    """Set up a simple logger"""
    console = StreamHandler()
    time_format = "%Y-%m-%d %H:%M:%S"
    console.setFormatter(
        Formatter(
            fmt=f"%(asctime)s %(levelname)s ({region_code}):  %(message)s",
            datefmt=time_format,
        )
    )

    log = getLogger("S1M")
    log.addHandler(console)
    log.setLevel(INFO)
    return log


def get_grid() -> gpd.GeoDataFrame:
    return (
        gpd.read_file(
            "https://raw.githubusercontent.com/digitalearthpacific/dep-grid/master/grid_pacific.geojson"
        )
        .astype({"tile_id": str, "country_code": str})
        .set_index(["tile_id", "country_code"], drop=False)
    )


class Sentinel1LoaderMixin(object):
    def __init__(
        self,
        only_search_descending: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.only_search_descending = only_search_descending

    def _get_items(self, area):
        query = {}
        if self.only_search_descending:
            query["sat:orbit_state"] = {"eq": "descending"}
        # Do the search
        item_collection = search_across_180(
            area, collections=["sentinel-1-rtc"], datetime=self.datetime, query=query
        )

        # Fix a few issues with STAC items
        # fix_bad_epsgs(item_collection)
        # item_collection = remove_bad_items(item_collection)

        if len(item_collection) == 0:
            raise EmptyCollectionError()

        return item_collection


class Sentinel1Loader(Sentinel1LoaderMixin, OdcLoaderMixin, StackXrLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class S1Processor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        load_data_before_writing: bool = True,
        drop_vars: list[str] = [],
    ) -> None:
        super().__init__(
            send_area_to_processor,
        )
        self.drop_vars = drop_vars
        self.load_data_before_writing = load_data_before_writing

    def process(self, input_data: DataArray) -> Dataset:
        arrays = []
        for band in ["vv", "vh"]:
            arrays.append(input_data[band].median("time").rename(f"median_{band}"))
            arrays.append(input_data[band].mean("time").rename(f"mean_{band}"))
            arrays.append(input_data[band].std("time").rename(f"std_{band}"))

        # Add count
        arrays.append(input_data["vv"].count("time").rename("count"))

        # Merge the arrays together into a Dataset with the names we want
        data = merge(arrays, compat="override")

        # Set nodata on all the outputs
        for band in data.data_vars:
            data[band].attrs["nodata"] = -32768

        output = set_stac_properties(input_data, data)

        if self.load_data_before_writing:
            output = output.compute()

        return output


def main(
    region_code: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    dataset_id: str = "mosaic",
    memory_limit_per_worker: str = "50GB",
    n_workers: int = 2,
    threads_per_worker: int = 32,
    xy_chunk_size: int = 4096,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    base_product = "s1"
    grid = get_grid()
    area = grid.loc[[region_code]]

    log = get_logger(region_code)
    log.info(f"Starting processing for {region_code}")

    itempath = DepItemPath(
        base_product, dataset_id, version, datetime, zero_pad_numbers=True
    )
    stac_document = itempath.stac_path(region_code)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and blob_exists(stac_document):
        log.info(f"Item already exists at {stac_document}")
        # This is an exit with success
        raise typer.Exit()

    loader = Sentinel1Loader(
        epsg=3832,
        datetime=datetime,
        dask_chunksize=dict(time=1, x=xy_chunk_size, y=xy_chunk_size),
        load_as_dataset=True,
        odc_load_kwargs=dict(
            fail_on_error=False,
            resolution=10,
            groupby="solar_day",
            bands=["vv", "vh"],
        ),
        nodata_value=-32768,
    )

    log.info("Configuring processor")
    processor = S1Processor()

    log.info("Configuring writer")
    writer = AzureDsWriter(
        itempath=itempath,
        overwrite=overwrite,
        convert_to_int16=False,
        extra_attrs=dict(dep_version=version),
        write_multithreaded=True,
    )

    runner = SimpleLoggingAreaTask(
        id=region_code,
        area=area,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=log,
    )

    with Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit_per_worker,
    ):
        try:
            paths = runner.run()
            log.info(f"Completed writing to {paths[-1]}")
        except EmptyCollectionError:
            log.warning("No data found for this tile.")
        except Exception as e:
            log.exception(f"Failed to process {region_code} with error: {e}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
