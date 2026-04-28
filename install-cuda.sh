source $HOME/.local/bin/env

uv venv --allow-existing 
uv pip install --upgrade pip

uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
uv pip install https://github.com/Dao-AILab/flash-attention.git --no-build-isolation

# Setup general packages
uv pip install pysrt pymkv2 pillow pytesseract rich langcodes lingua-language-detector colorama
uv pip install transformers accelerate

# Setup devel packages
uv pip install ruff pytest py7zr

uv pip install nbstripout
nbstripout --install

# Install sub-convert
uv pip install -e .