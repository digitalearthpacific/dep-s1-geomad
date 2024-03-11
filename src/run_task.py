import geopandas as gpd
import typer
from dask.distributed import Client
from dep_tools.azure import blob_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.loaders import OdcLoader

from dep_tools.namers import DepItemPath
from dep_tools.processors import Processor
from dep_tools.searchers import PystacSearcher
from dep_tools.stac_utils import set_stac_properties
from dep_tools.utils import get_logger

import boto3

# from dep_tools.task import SimpleLoggingAreaTask
from dep_tools.writers import AzureDsWriter, AwsDsCogWriter
from planetary_computer import sign_url
from typing_extensions import Annotated
from xarray import DataArray, Dataset, merge

from dep_tools.aws import object_exists


def get_tiles() -> gpd.GeoDataFrame:
    return (
        gpd.read_file(
            "https://raw.githubusercontent.com/digitalearthpacific/dep-grid/master/grid_pacific.geojson"
        )
        .astype({"tile_id": str, "country_code": str})
        .set_index(["tile_id", "country_code"], drop=False)
    )


def get_item_path(
    base_product: str, version: str, year: int, prefix: str
) -> DepItemPath:
    return DepItemPath(
        base_product,
        "mosaic",
        version,
        year,
        zero_pad_numbers=True,
        prefix=prefix,
    )


class S1Processor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        load_data: bool = False,
        drop_vars: list[str] = [],
    ) -> None:
        super().__init__(
            send_area_to_processor,
        )
        self.drop_vars = drop_vars
        self.load_data = load_data

    def process(self, input_data: DataArray) -> Dataset:
        arrays = []
        for band in ["vv", "vh"]:
            arrays.append(input_data[band].median("time").rename(f"median_{band}"))
            arrays.append(input_data[band].mean("time").rename(f"mean_{band}"))
            arrays.append(input_data[band].std("time").rename(f"std_{band}"))

        # Add count
        arrays.append(input_data["vv"].count("time").rename("count").astype("int16"))

        # Merge the arrays together into a Dataset with the names we want
        data = merge(arrays, compat="override")

        # Set nodata on all the outputs
        for band in data.data_vars:
            if band == "count":
                data[band].attrs["nodata"] = 0
            else:
                data[band].attrs["nodata"] = -32768

        output = set_stac_properties(input_data, data)

        if self.load_data:
            output = output.compute()

        return output


def main(
    tile_id: Annotated[str, typer.Option()],
    year: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    output_bucket: str = None,
    output_resolution: int = 10,
    memory_limit_per_worker: str = "50GB",
    n_workers: int = 2,
    threads_per_worker: int = 32,
    xy_chunk_size: int = 4096,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    base_product = "s1"
    tiles = get_tiles()
    area = tiles.loc[[tile_id]]

    log = get_logger(tile_id, "Sentinel-1-Mosaic")
    log.info(f"Starting processing version {version} for {year}")

    itempath = get_item_path(base_product, version, year, prefix="dep")

    stac_document = itempath.stac_path(tile_id)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite:
        already_done = False
        if output_bucket is None:
            # The Azure case
            already_done = blob_exists(stac_document)
        else:
            # The AWS case
            already_done = object_exists(output_bucket, stac_document)

        if already_done:
            log.info(f"Item already exists at {stac_document}")
            # This is an exit with success
            raise typer.Exit()

    # A searcher to find the data
    searcher = PystacSearcher(
        catalog="https://planetarycomputer.microsoft.com/api/stac/v1/",
        collections=["sentinel-1-rtc"],
        datetime=year,
        query={"sat:orbit_state": {"eq": "descending"}},
    )

    # A loader to load them
    loader = OdcLoader(
        crs=3832,
        resolution=output_resolution,
        bands=["vv", "vh"],
        groupby="solar_day",
        chunks=dict(time=1, x=xy_chunk_size, y=xy_chunk_size),
        fail_on_error=False,
        patch_url=sign_url,
        overwrite=overwrite,
    )

    # A processor to process them
    processor = S1Processor()

    # And a writer to bind them
    if output_bucket is None:
        log.info("Writing with Azure writer")
        writer = AzureDsWriter(
            itempath=itempath,
            overwrite=overwrite,
            convert_to_int16=False,
            extra_attrs=dict(dep_version=version),
            write_multithreaded=True,
            load_before_write=True,
        )
    else:
        log.info("Writing with AWS writer")
        client = boto3.client("s3")
        writer = AwsDsCogWriter(
            itempath=itempath,
            overwrite=overwrite,
            convert_to_int16=False,
            extra_attrs=dict(dep_version=version),
            write_multithreaded=True,
            bucket=output_bucket,
            client=client,
        )

    with Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit_per_worker,
    ):
        try:
            # Run the task
            items = searcher.search(area)
            log.info(f"Found {len(items)} items")

            data = loader.load(items, area)
            log.info(f"Found {len(data.time)} timesteps to load")

            output_data = processor.process(data)
            log.info(
                f"Processed data to shape {[output_data.sizes[d] for d in ['x', 'y']]}"
            )

            paths = writer.write(output_data, tile_id)
            if paths is not None:
                log.info(f"Completed writing to {paths[-1]}")
            else:
                log.warning("No paths returned from writer")

        except EmptyCollectionError:
            log.warning("No data found for this tile.")
        except Exception as e:
            log.exception(f"Failed to process {tile_id} with error: {e}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
