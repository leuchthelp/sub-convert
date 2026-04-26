# sub-convert

![](files/example.png)

**WARNING** current only tested for AMD GPUs and CPU, needs testing for other vendors, see the [Installation guide](quick-start#installation-guide) 

sub-convert is a simple project inspired by [pgsrip](https://github.com/ratoaq2/pgsrip) by [ratoaq2](https://github.com/ratoaq2). It is meant to convert PGS (image-based) subtitles to SRT (text-based) subtitles using a shared OCR model which `N` processes can request `image-to-text` conversion from. 

Please refer to the [current roadmap](#current-roadmap) for information on future development.

It tries to overcome some of the key [shortcomings](#shortcomings-include) of pgsrip. However some parts of pgsrip have been retained, more specifically the PGS parser build by [ratoaq2](https://github.com/ratoaq2), which as since been improved to be feature complete with [this documentation](https://blog.thescorpius.com/index.php/2017/07/15/presentation-graphic-stream-sup-files-bluray-subtitle-format/). 

## Introduction


## Shortcomings include:

- PGS parser not being fully featured, skipping images entirely
- overlapping PGS subtitles not being converted correctly or skipped entirely
- handling off fade ins / outs within PGS
- the use of tesseract for OCR
- images internally not being extracted correct (styling, alpha missing)
- handling of forced subtitles
- handling of subtitles with mislabeled languages by the manufacturer
- the handling of final file path naming (current approach isn't the best either)
- parallelism for multiple files

For more information on PGS, the issue encountered & the solution provided, please visit the [portion of the documentation focussing on PGS](pgs.md) more directly.

## To fix these issues the following conceptual changes have been applied:

- keep pure `CPU` support with [pytesseract](https://github.com/madmaze/pytesseract) & [lingua-py](https://github.com/pemistahl/lingua-py) 
- add [PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) as the main OCR, tesseract exists as a fallback
- add [Mike0307/multilingual-e5-language-detection](https://huggingface.co/Mike0307/multilingual-e5-language-detection) for language detection
- properly handle [Fades (In / Out)](pgs.md#fades-in--out), [Overlaps](pgs.md#overlaps), empty images. etc
- assume subtitles is forced if less than 150 subtitles items are with the track, otherwise set flag if set in original file
- parallelism via `multiprocessing`, 4 different subtitles track will be converted at a time (can be configured)

## Caveats

Tools like [Subtitle Edit](https://www.nikse.dk/subtitleedit) do exists, which will always be more accurate and stable due to the sheer amount of work already poured into the project. 

However [Subtitle Edit](https://www.nikse.dk/subtitleedit) does not offer a similar CLI and has resulted in less accurate conversions, while not handling fades at all.

## Benefits

This approaches aims to provide a middle group for user that can live with the occasional misidentified character in their subtitles and would like the benefit from the "hands-off" approach to conversion.

The `ModelCore` is designed to be extendable so future models can be swapped easily for better overall recognition. 

## Current roadmap

The current plan is to design a tool than can handle all kinds of `CPU`, `GPU`, etc. combinations and is quick and easy to install. (This requires the underlying models to behave nicely)

### Jellyfin plugin interface

Since Docker containers will most likely be the intended way of interacting with this project, I thought up the concept of creating a custom jellyfin plugin that can interface with it. This is an early stage idea, but I image a server running inside the container waiting for a request from jellyfin which points to a directory path were new `.mkv` files have been added. 

It will then launch `sub-convert` and tell it to look for files in the requested path and convert their contents.

### More formats to support

As this approach takes `images` and converts them to `text` any image-based subtitles format could be converted to any text-based subtitle format.

Current on PGS (image-based) and SRT (text-based) is supported. However, If I can get my hands on a functioning parser than can spit out image data, it can be used with this approach.

Similarly I would like to take advantage of [ASS](http://www.tcax.org/docs/ass-specs.htm) subtitles more expressive options, which could retain original PGS subtitles text colors, positioning and so on. I am simply not knowledgeable enough on them yet to properly implement a conversion.

### GPU support

There already is a problem with this approach for AMD GPUs. They do not work out of the box as installing the required software-stack with proper [rocm](https://rocm.docs.amd.com/en/latest/index.html) support. 

### Docker support

To isolate different software stacks and to add proper support Dockerfile will be the main way to go for this project. Since I only own an AMD GPU, this is the only use-case I can test. 

Work is underway to unify the Dockerfile for all platforms targeted.