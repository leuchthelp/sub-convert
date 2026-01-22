FROM rocm/pytorch:latest


RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
RUN update-locale
RUN apt-get update && apt-get install -y tesseract-ocr mkvtoolnix 

RUN . /opt/venv/bin/activate
RUN git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install