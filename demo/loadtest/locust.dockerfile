FROM locustio/locust

USER root

# Install prometheus_client and other dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install prometheus_client
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /mnt/locust

USER locust
