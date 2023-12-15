import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def check_gpu_availability():
    console = Console()
    
    if torch.cuda.is_available():
        pytorch_version = torch.__version__
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(0)  # Sets the default GPU as GPU 0

        # Creating rich text for output
        gpu_info_text = Text(f"CUDA (GPU support) is available on this device.\n", style="bold green")
        gpu_info_text.append(f"PyTorch version: {pytorch_version}\n")
        gpu_info_text.append(f"Number of GPUs available: {num_gpus}\n", style="bold blue")

        for i in range(num_gpus):
            gpu_info_text.append(f"GPU {i}: {torch.cuda.get_device_name(i)}\n", style="yellow")

        current_device = torch.cuda.current_device()
        gpu_info_text.append(f"Current CUDA device index: {current_device}", style="bold magenta")

        # Displaying information in a panel
        console.print(Panel(gpu_info_text, title="GPU Information", expand=False, style="bold purple"))
    else:
        console.print("CUDA (GPU support) is not available on this device.", style="bold red")

