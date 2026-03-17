import time
import torch


def hold_vram_on_device(device, target_ratio=0.9):
    props = torch.cuda.get_device_properties(device)
    total_mem = props.total_memory
    target_bytes = int(total_mem * target_ratio)

    # 留一点余量，减少因碎片或上下文导致的 OOM 风险
    chunk_bytes = 256 * 1024 * 1024  # 256 MB
    num_chunks = target_bytes // chunk_bytes

    tensors = []
    allocated = 0

    for _ in range(num_chunks):
        n = chunk_bytes // 4  # float32 = 4 bytes
        t = torch.empty(n, dtype=torch.float32, device=f"cuda:{device}")
        tensors.append(t)
        allocated += t.numel() * t.element_size()

    remaining = target_bytes - allocated
    if remaining > 0:
        n = remaining // 4
        if n > 0:
            t = torch.empty(n, dtype=torch.float32, device=f"cuda:{device}")
            tensors.append(t)
            allocated += t.numel() * t.element_size()

    return {
        "device": device,
        "name": props.name,
        "total_gb": total_mem / 1024**3,
        "held_gb": allocated / 1024**3,
        "tensors": tensors,
    }


def hold_vram_multi(devices=None, target_ratio=0.9, duration_sec=60):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    gpu_count = torch.cuda.device_count()

    if devices is None:
        devices = list(range(gpu_count))

    for d in devices:
        if d < 0 or d >= gpu_count:
            raise ValueError(f"Invalid GPU id: {d}, available range: 0..{gpu_count - 1}")

    holders = []

    try:
        print(f"Using GPUs: {devices}")
        print(f"Target ratio per GPU: {target_ratio:.0%}")
        print(f"Duration: {duration_sec} seconds")
        print("-" * 50)

        for d in devices:
            info = hold_vram_on_device(d, target_ratio=target_ratio)
            holders.append(info)
            print(
                f"GPU {info['device']} | {info['name']} | "
                f"Total: {info['total_gb']:.2f} GB | "
                f"Held: {info['held_gb']:.2f} GB"
            )

        print("-" * 50)
        print("Holding memory...")
        time.sleep(duration_sec)

    finally:
        print("Releasing memory...")
        for h in holders:
            h["tensors"].clear()
        torch.cuda.empty_cache()
        print("Done.")


if __name__ == "__main__":
    # 例子：同时占用 0、1、2 号 GPU 各 90% 显存，持续 60 秒
    hold_vram_multi(devices=[0, 1], target_ratio=0.9, duration_sec=60*60*24)