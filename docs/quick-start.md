# Quick-start

## Installation guide

Requires `python >= 3.12`

The project currently interfaces with the models through huggingfaces [transformers](https://huggingface.co/docs/transformers/index).

First install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

In the future the project will be entirely switched over to uv, however due to various dependency conflicts with pytorch, rocm, flash_attn and alike, simple install script are the way to go. Until I can figure out a proper, native uv project, there will be no build for this project.

There are five install script denoted with their respective platform:

```bash
-rocm
-cuda
-openvino
-base
-macos
```

Simply execute the matching script with `bash install-{choice}.sh`

[flash-attention](https://github.com/Dao-AILab/flash-attention) is optional but will be installed for the [rocm](install-rocm.sh) & [cuda](install-cuda.sh), as it has been validated to work.

If you do not install flash_attention, the tool will fallback to pytorches integrated [sdpa](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) attention backend, which should work on all platforms.

## Usage

The script provides progress bars for each cpu worker launched. If the progressbar shows a stalled process it is most like a visual bug with `rich` the process will have finished if overall progress bars for `N = number of cpu workers` are displayed.

You interact with the tool via cli, like:

```bash
sub-convert -p test-files

or

uv run sub-convert -p test-files
```

To optimize on resource usage the tool utilizes generators and draws files to convert lazily when needed. 

This however, results in the tool not being able to tell how many files it will convert in total and if it has converted any to begin with. If it exists without an error while not showing a single progressbar, it most likely has not found any files to convert. In that case make sure to check if the path you have given sub-convert is accessible.


When files are being saved, existing files can also be override by specifying:

```bash
sub-convert -o

or 

uv run sub-convert -o
```

You can switch the type of OCR or language model-core you want to use by supplying `-om` or `-lm`. 

To see the cores available simply run `sub-convert --help` and they will be listed for both options.

Using `-a` lets you define a point in time after or before which all existing `.srt` files will be replaced and their original `.mkv` will be processed. All files outside this range will be skipped. 

```bash
-a d+1
```
means: all files older than 1 day will be processed, younger files will be skipped

```bash
-a w-1 #possible: ms, s, m (minutes), d, h, w, M (months), y 
```
means: all files younger than 1 week will be processed, older files will be skipped


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

Additionally the `-b, --batchsize` arguments exists to batch images for inference, however, this options has not been tested much due to AMD GPU crashes for `rocm/pytorch` docker containers - use with caution.

Lastly, `-d` dumps debug files like - DisplaySet, all associated images, TimelineItems exported as images, TimelineItems exported as Pandas Dataframe in JSON. 
Ploty & Kaleido need to be installed for this, as well as any version of Google Chrome. Otherwise plotly will be unabled to export an image visualization of the TimelineItems as `.svg`.