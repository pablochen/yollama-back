FROM ubuntu:22.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

LABEL maintainer="pablochen <bogenarc@gmail>"

ENV PYTHON_VERSION=3.10

RUN apt-get update && \
    apt-get install -y python${PYTHON_VERSION} && \
	apt-get install -y python3-pip && \
	apt-get install -y git && \
	apt-get install -y cmake && \
	apt-get install -y build-essential && \
	apt-get clean


WORKDIR /app

RUN \
   echo 'alias python="/usr/bin/python3"' >> /root/.bashrc && \
   echo 'alias pip="/usr/bin/pip3"' >> /root/.bashrc && \
   source /root/.bashrc


RUN git clone https://github.com/unslothai/unsloth.git && \
    cd unsloth && \
    pip install ".[colab-new]"

RUN pip install packaging

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN rm -rf /app/requirements.txt /app/Dockerfile

ENV PYTHONUNBUFFERED=1
  
CMD ["uvicorn", "back:app", "--host", "0.0.0.0", "--port", "8000"]