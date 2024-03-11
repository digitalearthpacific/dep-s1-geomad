build:
	docker build --tag dep/s1 .

run:
	docker run -it --rm \
		-e PC_SDK_SUBSCRIPTION_KEY=${PC_SDK_SUBSCRIPTION_KEY} \
		-e AZURE_STORAGE_ACCOUNT="deppcpublicstorage" \
		-e AZURE_STORAGE_SAS_TOKEN=${AZURE_STORAGE_SAS_TOKEN} \
	 	dep/s1 \
		python src/run_task.py \
		--region-code "64,20" \
		--datetime 2023 \
		--version "0.0.0t0" \
		--resolution 100 \
		--output-bucket files.auspatious.com \
		--overwrite
