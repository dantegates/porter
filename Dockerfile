FROM python:2.7

ADD requirements.txt .
RUN python -m pip install -r requirements.txt

# Add ipa to the standard directory for custom packages so that users
# can import it from anywhere
# https://docs.python.org/2/library/site.html#site.USER_SITE
ADD ipa /root/.local/lib/python2.7/site-packages/ipa/
ENV PYTHONPATH=/root/.local/lib/python2.7/site-packages/ipa:$PYTHONPATH

ADD tests ./tests
ADD runtests.sh .
RUN ./runtests.sh

WORKDIR /code

ENTRYPOINT ["python"]
