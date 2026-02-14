# sub-convert

**WARNING** current only tested for AMD GPUs and CPU, needs testing for other vendors, see the [Installation guide](#installation-guide) 

sub-convert is a simple project inspired by [pgsrip](https://github.com/ratoaq2/pgsrip) by [ratoaq2](https://github.com/ratoaq2). It is meant to convert PGS (image-based) subtitles to SRT (text-based) subtitles using a shared OCR model which `N` processes can request `image-to-text` conversion from. 

Please refer to the [current roadmap](#current-roadmap) for information on future development.

It tries to overcome some of the key [shortcomings](#shortcomings-include) of pgsrip. However some parts of pgsrip have been retained, more specifically the PGS parser build by [ratoaq2](https://github.com/ratoaq2). 

## Installation guide

Requires `python >= 3.12`

Since I could not get the software stack to behave properly on my AMD GPU development has been done inside of a Docker Container. CPU usage should work bare metal but anything else is up to sheer luck.

The project currently interfaces with the models through huggingfaces [transformers](https://huggingface.co/docs/transformers/index). However I am interest in learning [vllm](https://docs.vllm.ai/en/latest/) and switching to a fully server based architecture internally or using the tools on bare metal like with [paddleOCR](https://pypi.org/project/paddleocr/), which is not compatible with AMD GPUs at the moment.

For AMD user, please use the [Dockerfile-rocm](Dockerfile-rocm) dockerfile. For active development a rocm compatible `devcontainer` is also available.

NVIDIA & CPU users could try to just install the project on bare metal and see how it goes. For this purpose install the dependencies from:

```py
pip install -r requirements-external.txt
```

`requirements.txt` is meant to be used inside a [pytorch container](https://hub.docker.com/r/pytorch/pytorch/). 

[flash-attention](https://github.com/Dao-AILab/flash-attention) is optional but comes already installed in the [rocm](Dockerfile-rocm) container, as it has been validate to work. NVIDIA user might also like to installed this additional dependencies for potentially better performance.

If you do not install flash_attention, the tool will fallback to pytorches integrated [sdpa](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) attention backend, which should work on all platforms.

## Usage

Either you interact with the [main.py](main.py) script directly and change `root = Path("test-files")` inside the `main()` to a directory of your choice. 

Or you interact with the tool via cli, like:

```bash
python main.py -p test-files
```

When files are being saved, existing files can also be override by specifying:

```bash
python main.py -o True
```

or by changing:

```py
options = {
        "path_to_tmp": "tmp",
        "override_if_exists": args.override
    }
```
inside the `main()` of the script. 

Using `-s True` will skip a files for which subtitle already exists. Due to the fact that naming cannot be inferred back to the tracks within a file no track will be processed even if the subtitles found only belong to one of multiple tracks in the `MKV` file.

The current architecture allows you to launch `N` OCR model gpu workers followed by `N` language model gpu workers. `N=4` cpu workers each work on a single subtitle track for which `pgs images` corresponding to the amount of images found in the track are processed. Each image instance is processed one-by-one. 

The amount of workers can be adjusted with the following arguments:

```console
-c, --cpu_workers N
-ow, --ocr_workers N
-lw, --lang_workers N
```

Additionally the `-b, --batchsize` arguments exists to batch imaged for inference, however, this options has not been tested much due to AMD gpu crashes - use with caution.

## Shortcomings include:

- the use of tesseract for OCR
- handling of forced subtitles
- handling of subtitles with mislabeled languages by the manufacturer
- the handling of final file path naming (current approach isn't the best either)
- parallelism for multiple files (should already be possible but I couldn't get my system saturated to satisfaction)

## To fix these issues the following conceptual changes have been applied:

- add [PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) as the main OCR, tesseract exists as a fallback
- add [Mike0307/multilingual-e5-language-detection](https://huggingface.co/Mike0307/multilingual-e5-language-detection) for language detection
- assume subtitles is forced if less than 150 subtitles items are with the track, otherwise set flag if set in original file
- parallelism via `multiprocessing`, 4 different subtitles track will be converted at a time (can be configured)

## Caveats

Going with a traditional tesseract approach is better in 90% of cases. Tesseract is much fast and requires less resources in terms of RAM, VRAM or additional GPUs. As such this approach will require more resources and take longer for full scans of large libraries. Be wary.

Additionally tools like [Subtitle Edit](https://www.nikse.dk/subtitleedit) do exists, which will always be more accurate and stable due to the sheer amount of work already poured into the project.

## Benefits

This approaches aims to provide a middle group for user that can live with the occasional misidentified character in their subtitles and would like the benefit from the "hands-off" approach to conversion.

The `ModelCore` is designed to be extendable so future models can be swapped easily for better overall recognition. 

## Current roadmap

The current plan is to design a tool than can handle all kinds of `CPU`, `GPU`, etc. combinations and is quick and easy to install. (This requires the underlying models to behave nicely)

### Jellyfin plugin interface

Since Docker containers will most likely be the intended way of interacting with this project, I thought up the concept of creating a custom Jellyfin plugin that can interface with it. This is an early stage idea, but I image a server running inside the container waiting for a request from jellyfin which points to a directory path were new `.mkv` files have been added. 

It will then launch `sub-convert` and tell it to look for files in the requested path and convert their contents.

### More formats to support

As this approach takes `images` and converts them to `text` any image-based subtitles format could be converted to any text-based subtitle format.

Current on PGS (image-based) and SRT (text-based) is supported. However, If I can get my hands on a functioning parser than can spit out image data, it can be used with this approach.

Similarly I would like to take advantage of [ASS](http://www.tcax.org/docs/ass-specs.htm) subtitles more expressive options, which could retain original PGS subtitles text colors, positioning and so on. I am simply not knowledgeable enough on them yet to properly implement a conversion.

### GPU support

There already is a problem with this approach for AMD GPUs. They do not work out of the box as installing the required software-stack with proper [rocm](https://rocm.docs.amd.com/en/latest/index.html) support. Running rocm in docker is much easier. Therefore it is recommended to use the provided [rocm Dockerfile](Dockerfile-rocm)

### Docker support

To isolate different software stacks and to add proper support Dockerfile will be the main way to go for this project. Since I only own an AMD GPU, this is the only use-case I can test. 

However I will simply assume a regular pytorch container will just work for CUDA, since ... you know ... CUDA.

Intel will be a little different, as I do not have any experience in that regard. If you have an Intel GPU and could test if you can get an environment up and running, then please feel free to create a pull request.
