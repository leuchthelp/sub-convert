source $HOME/.local/bin/env

uv venv --allow-existing 
uv pip install --upgrade pip

uv pip install openvino
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/xpu

# Setup general packages
uv pip install pysrt pymkv2 pillow pytesseract rich langcodes lingua-language-detector colorama
uv pip install transformers accelerate

# Setup devel packages
uv pip install ruff pytest py7zr

uv pip install nbstripout
nbstripout --install

# Install sub-convert
uv pip install -e .