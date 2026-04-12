FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.14_pytorch_2.10.0


RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
RUN update-locale
RUN apt-get update && apt-get install -y tesseract-ocr mkvtoolnix 

RUN apt-get install -y wget
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \ 
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list
RUN apt-get update && apt-get -y install google-chrome-stable

RUN pip install --upgrade pip
RUN pip install pymkv2 pytesseract pysrt pillow rich langcodes colorama einops transformers protobuf sentencepiece plotly kaleido nbformat

ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
RUN . /opt/venv/bin/activate
RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
    cd flash-attention &&\
    python setup.py install