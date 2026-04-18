# sub-convert

**WARNING** current only tested for AMD GPUs and CPU, needs testing for other vendors, see the [Installation guide](#installation-guide) 

sub-convert is a simple project inspired by [pgsrip](https://github.com/ratoaq2/pgsrip) by [ratoaq2](https://github.com/ratoaq2). It is meant to convert PGS (image-based) subtitles to SRT (text-based) subtitles using a shared OCR model which `N` processes can request `image-to-text` conversion from. 

Please refer to the [current roadmap](#current-roadmap) for information on future development.

It tries to overcome some of the key [shortcomings](#shortcomings-include) of pgsrip. However some parts of pgsrip have been retained, more specifically the PGS parser build by [ratoaq2](https://github.com/ratoaq2). 

## Installation guide

Requires `python >= 3.12`

Since I could not get the software stack to behave properly on my AMD GPU development has been done inside of a Docker Container. CPU usage should work bare metal but anything else is up to sheer luck.

The project currently interfaces with the models through huggingfaces [transformers](https://huggingface.co/docs/transformers/index).

For AMD user, please use the [Dockerfile-rocm](Dockerfile-rocm) dockerfile. For active development a rocm compatible `devcontainer` is also available.

NVIDIA & CPU users could try to just install the project on bare metal and see how it goes. For this purpose install the dependencies from:

```py
pip install -r requirements-external.txt
```

`requirements.txt` is meant to be used inside a [pytorch container](https://hub.docker.com/r/pytorch/pytorch/). 

[flash-attention](https://github.com/Dao-AILab/flash-attention) is optional but comes already installed in the [rocm](Dockerfile-rocm) container, as it has been validated to work. NVIDIA user might also like to installed this additional dependencies for potentially better performance.

If you do not install flash_attention, the tool will fallback to pytorches integrated [sdpa](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) attention backend, which should work on all platforms.

## Usage

The script provides progress bars for each cpu worker launched. If the progressbar shows a stalled process it is most like a visual bug with `rich` the process will have finished if overall progress bars for `N = number of cpu workers` are displayed.

You interact with the tool via cli, like:

```bash
python main.py -p test-files
```

When files are being saved, existing files can also be override by specifying:

```bash
python main.py -o
```

Using `-s` will skip files for which subtitles already exist. Due to the fact that naming cannot be inferred back to the tracks within a file no track will be processed even if the subtitles found only belong to one of multiple tracks in the `MKV` file.

The current architecture allows you to launch `N` OCR model GPU workers followed by `N` language model GPU workers. `N=4` CPU workers each work on a single subtitle track for which `pgs images` corresponding to the amount of images found in the track are processed. Each image instance is processed one-by-one. 

Each worker is launches as a separate process meaning you will need at least `N_cw + N_ow + N_lw + 2` threads available on your system. The default is 6 threads meaning a 3 core CPU with 2 threads per core is required at the very least. The extra `+2` are Managers with handle communication between processes via `Queues`. One manager controls the GPU queues, while the other controls the CPU and progress queues (used for progress bar). 

All CPU workers queue their images towards a global GPU queue. OCR GPU workers than draw items from the first queue and processes the images. Once processed the extracted text is passed through another queue towards the language model workers which classify the language of the text. 

Finally the language model workers send the text with the language classification back to the CPU worker who initially processes this item, ensuring processed tracks remain consistent and ordered.

The amount of workers can be adjusted with the following arguments:

```console
-c, --cpu_workers N
-ow, --ocr_workers N
-lw, --lang_workers N
```

Additionally the `-b, --batchsize` arguments exists to batch images for inference, however, this options has not been tested much due to AMD GPU crashes - use with caution.
