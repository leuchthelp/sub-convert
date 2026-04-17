import os

from torch import cuda, version, xpu


def check_torch_cuda(options: dict[str, str | bool]) -> dict[str, str | bool]:
    # Setup basic options relating to pytorch and set environmental variables if needed
    torch_device = "cuda" if cuda.is_available() else "cpu"
    if torch_device == "cuda":
        # Check for working rocm and activate flash attention, otherwise its NVIDIA
        if version.hip is not None:
            os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
            # os.environ["FLASH_ATTENTION_TRITON_AMD_AUTOTUNE"] = "TRUE"

    if xpu.is_available():
        options["intel_disable_flash"] = True
        torch_device = "xpu"

    options["torch_device"] = torch_device

    check = options
    return check
