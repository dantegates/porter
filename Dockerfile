FROM python:3.6

ARG PORTER_S3_ACCESS_KEY_ID
ARG PORTER_S3_SECRET_ACCESS_KEY
ARG PORTER_S3_BUCKET_TEST

WORKDIR /porter

ADD setup.py .
RUN python3.6 -m pip install .[keras-utils,sklearn-utils,s3-utils]
ADD porter ./porter
RUN python3.6 setup.py install

ADD tests ./tests
ADD runtests.sh .

RUN ./runtests.sh

WORKDIR /code

ENTRYPOINT ["python3.6"]
