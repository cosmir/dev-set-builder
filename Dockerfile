FROM ubuntu:latest
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
ENTRYPOINT ["python3"]
RUN apt-get update && \
  apt-get install -y git-all wget
COPY ./ /dev-set-builder
WORKDIR dev-set-builder/
RUN scripts/download-deps.sh
RUN pip install -e .