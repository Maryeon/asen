FROM  pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

RUN apt update
RUN apt install -y wget libglib2.0-0 libsm6 libxext6 libxrender1

COPY requirements.txt /requrements.txt
RUN pip install -r /requrements.txt

