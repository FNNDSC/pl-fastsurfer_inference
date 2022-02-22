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
# In the case of a proxy (located at 192.168.13.14:3128), do:
#
#    docker build --build-arg http_proxy=http://192.168.13.14:3128 --build-arg UID=$UID -t local/pl-fastsurfer_inference .
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
LABEL maintainer="Martin Reuter (FastSurfer), Sandip Samal (FNNDSC) (sandip.samal@childrens.harvard.edu) <dev@babyMRI.org>"

WORKDIR /usr/local/src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD ["fastsurfer_inference", "--help"]
