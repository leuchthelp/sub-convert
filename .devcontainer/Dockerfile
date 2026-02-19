FROM rocm/pytorch:latest


RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
RUN update-locale
RUN apt-get update && apt-get install -y tesseract-ocr mkvtoolnix 

RUN pip install --upgrade pip
RUN pip install pymkv2 pytesseract pysrt pillow rich langcodes colorama einops transformers==4.57.6 protobuf sentencepiece

ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
RUN . /opt/venv/bin/activate
RUN git clone https://github.com/Dao-AILab/flash-attention &&\ 
    cd flash-attention &&\
    python setup.py install