# Docker file for fastsurfer_inference ChRIS plugin app
#
# Build with
#
#   docker build -t <name> .
#
# For example if building a local version, you could do:
#
#   docker build -t local/pl-fastsurfer_inference .
#
# In the case of a proxy (located at say 10.41.13.4:3128), do:
#
#    export PROXY="http://10.41.13.4:3128"
#    docker build --build-arg http_proxy=${PROXY} --build-arg UID=$UID -t local/pl-fastsurfer_inference
#
# To run an interactive shell inside this container, do:
#
#   docker run -ti --entrypoint /bin/bash local/pl-fastsurfer_inference
#
# To pass an env var HOST_IP to container, do:
#
#   docker run -ti -e HOST_IP=$(ip route | grep -v docker | awk '{if(NF==11) print $9}') --entrypoint /bin/bash local/pl-fastsurfer_inference
#



FROM tensorflow/tensorflow:latest-gpu-py3
ARG PYTHON_VERSION=3.6
LABEL MAINTAINER="dev@babymri.org"

ENV APPROOT="/usr/src/fastsurfer_inference"
COPY ["fastsurfer_inference", "${APPROOT}"]
COPY ["requirements.txt", "${APPROOT}"]
COPY ["checkpoints", "/usr/src/checkpoints"]

WORKDIR $APPROOT
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["fastsurfer_inference.py", "--help"]
