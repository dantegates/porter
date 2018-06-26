make build:
	docker build \
		--build-arg PORTER_S3_ACCESS_KEY_ID=$PORTER_S3_ACCESS_KEY_ID \
		--build-arg PORTER_S3_SECRET_ACCESS_KEY=$PORTER_S3_SECRET_ACCESS_KEY \
		--build-arg PORTER_S3_BUCKET_TEST=$PORTER_S3_BUCKET_TEST \
		-t ${ARGS} .

make test:
	python3.6 -m unittest discover -s tests
