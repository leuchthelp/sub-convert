apt-get clean && apt-get update 
apt-get update && apt-get install -y tesseract-ocr mkvtoolnix 

source $HOME/.local/bin/env

uv venv --allow-existing 
uv pip install --upgrade pip


# Setup AMD specific packages
uv pip install https://github.com/ROCm/aiter.git
uv pip install https://github.com/ROCm/iris.git

uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.2
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" uv pip install https://github.com/ROCm/flash-attention.git --no-build-isolation

# Setup general packages
uv pip install pysrt pymkv2 pillow pytesseract rich langcodes lingua-language-detector
uv pip install transformers accelerate

# Setup devel packages
uv pip install ruff pytest py7zr


# Install sub-convert
uv pip install -e .