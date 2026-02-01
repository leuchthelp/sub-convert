FROM rocm/pytorch:latest


RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
RUN update-locale
RUN apt-get update && apt-get install -y tesseract-ocr mkvtoolnix 

ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
RUN . /opt/venv/bin/activate
RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
    cd flash-attention &&\
    python setup.py install