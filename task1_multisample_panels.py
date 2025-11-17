from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# Base directories
PRED_ROOT = Path("/labs/Yu/Jahan/predictions")
FIG_DIR = Path("/home/FCAM/jghasemi/projects/ctxseg/figures_task1")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Final multi-ID configuration
CONFIG = {
    "livecell": {
        "image_ids": ["100163", "100450", "100601", "100785", "98960"]
    },
    "tissuenet": {
        "image_ids": ["0", "1000", "1001", "1002", "1003"]
    },
    "nips": {
        "image_ids": ["0", "10", "11", "12", "13"]
    },
}

def load_image_2d(path: Path):
    arr = tiff.imread(path)
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr

def load_pred_stack(path: Path):
    arr = tiff.imread(path)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr[None, ...]

    if arr.ndim == 3:
        sample_axis = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, sample_axis, 0)
        return arr

    if arr.ndim == 4:
        sample_axis = None
        for ax, size in enumerate(arr.shape):
            if size in (4, 8, 16):
                sample_axis = ax
                break
        if sample_axis is None:
            sample_axis = int(np.argmin(arr.shape))

        arr = np.moveaxis(arr, sample_axis, 0)

        if arr.ndim == 4:
            if arr.shape[1] <= 4:
                arr = arr[:, 0, ...]
            else:
                arr = arr[..., 0]

        return arr

    raise ValueError(f"Unexpected pred shape: {arr.shape}")

def build_panel(dataset: str, image_id: str, max_samples: int = 5):
    diff_dir = PRED_ROOT / "diffusion_results" / dataset
    base_dir = PRED_ROOT / "baseline" / dataset

    input_path  = diff_dir / f"{image_id}.tif"
    label_path  = diff_dir / f"{image_id}_label.tif"
    pred_path   = diff_dir / f"{image_id}_pred.tif"
    diff_out    = diff_dir / f"{image_id}_output.tif"
    base_out    = base_dir / f"{image_id}_output.tif"

    for p in [input_path, label_path, pred_path, diff_out, base_out]:
        if not p.exists():
            print(f"[ERROR] Missing file: {p}")
            return

    img = load_image_2d(input_path)
    gt = load_image_2d(label_path)
    pred_stack = load_pred_stack(pred_path)
    diff_img = load_image_2d(diff_out)
    base_img = load_image_2d(base_out)

    n_samples = pred_stack.shape[0]
    n_use = min(max_samples, n_samples)

    print(f"[INFO] {dataset}/{image_id}: slices={n_samples}, using={n_use}")

    n_cols = 2 + n_use + 2
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3))

    col = 0

    axes[col].imshow(img, cmap="gray")
    axes[col].set_title("Input")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(gt, cmap="nipy_spectral")
    axes[col].set_title("GT")
    axes[col].axis("off")
    col += 1

    for i in range(n_use):
        axes[col].imshow(pred_stack[i], cmap="nipy_spectral")
        axes[col].set_title(f"Sample {i+1}")
        axes[col].axis("off")
        col += 1

    axes[col].imshow(diff_img, cmap="nipy_spectral")
    axes[col].set_title("Diffusion Out")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(base_img, cmap="nipy_spectral")
    axes[col].set_title("Baseline Out")
    axes[col].axis("off")

    fig.suptitle(f"{dataset} – ID {image_id}", y=0.97)
    plt.tight_layout()

    out_file = FIG_DIR / f"{dataset}_fullpanel_{image_id}.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {out_file}")

def main():
    for dataset, cfg in CONFIG.items():
        for image_id in cfg["image_ids"]:
            build_panel(dataset, image_id, max_samples=5)

if __name__ == "__main__":
    main()
