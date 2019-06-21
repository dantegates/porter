build:
	docker build \
		--build-arg PORTER_S3_ACCESS_KEY_ID=$PORTER_S3_ACCESS_KEY_ID \
		--build-arg PORTER_S3_SECRET_ACCESS_KEY=$PORTER_S3_SECRET_ACCESS_KEY \
		--build-arg PORTER_S3_BUCKET_TEST=$PORTER_S3_BUCKET_TEST \
		-t ${ARGS} .

test:
	python3.6 -m unittest discover -s tests

lint:
	python3.6 -m pylint --errors-only porter

coverage:
	coverage run --source porter -m unittest discover -s tests
	coverage html
	open htmlcov/index.html

install:
	python3.6 -m pip install .[keras-utils,sklearn-utils,s3-utils]

docs:
	docker run \
	    -v $(shell pwd)/docs:/local \
	    --entrypoint docker-entrypoint.sh \
	    openapitools/openapi-generator-cli \
	    generate -i /local/porter-api.yaml -g html -o /local/html --skip-validate-spec --generate-alias-as-model

.PHONY: docs
