from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

# Base directories
PRED_ROOT = Path("/labs/Yu/Jahan/predictions")
FIG_DIR = Path("/home/FCAM/jghasemi/projects/ctxseg/figures_task1")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Configure the image IDs for each dataset
CONFIG = {
    "livecell": {
        "image_id": "98960",
    },
    # Add more after livecell works:
    # "tissuenet": {"image_id": "XXXXX"},
    # "nips": {"image_id": "YYYYY"},
}

def load_image_2d(path: Path):
    """Load a TIFF as 2D (H, W)."""
    arr = tiff.imread(path)
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr

def load_pred_stack(path: Path):
    """
    Load prediction stack and return shape (N, H, W).
    Handles shapes like:
      - (H, W)
      - (N, H, W)
      - (H, W, N)
      - (C, N, H, W)
      - (N, C, H, W)
      - (C, H, W, N)
      - etc.
    """
    arr = tiff.imread(path)
    arr = np.squeeze(arr)

    # Case 2D: single mask
    if arr.ndim == 2:
        return arr[None, ...]

    # Case 3D: treat smallest axis as sample axis
    if arr.ndim == 3:
        sample_axis = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, sample_axis, 0)
        return arr

    # Case 4D: bring sample axis to front, collapse channels
    if arr.ndim == 4:

        # First try to find a reasonable sample axis (usually small size like 4, 8, 16)
        sample_axis = None
        for ax, size in enumerate(arr.shape):
            if size in (4, 8, 16):
                sample_axis = ax
                break

        # If not found, fallback to smallest dimension
        if sample_axis is None:
            sample_axis = int(np.argmin(arr.shape))

        # Move sample axis to front -> (N, *, *, *)
        arr = np.moveaxis(arr, sample_axis, 0)

        # Collapse channels if still 4D
        if arr.ndim == 4:
            # (N, C, H, W)
            if arr.shape[1] <= 4:
                arr = arr[:, 0, ...]
            else:
                # (N, H, W, C)
                arr = arr[..., 0]

        return arr

    raise ValueError(f"Unexpected pred shape: {arr.shape}")

def build_panel(dataset: str, cfg: dict, max_samples: int = 5):
    image_id = cfg["image_id"]

    # Paths for diffusion model
    diff_dir = PRED_ROOT / "diffusion_results" / dataset
    input_path = diff_dir / f"{image_id}.tif"
    label_path = diff_dir / f"{image_id}_label.tif"
    pred_stack_path = diff_dir / f"{image_id}_pred.tif"
    diff_output_path = diff_dir / f"{image_id}_output.tif"

    # Paths for baseline model
    baseline_dir = PRED_ROOT / "baseline" / dataset
    baseline_output_path = baseline_dir / f"{image_id}_output.tif"

    # Sanity check files
    required = [
        input_path,
        label_path,
        pred_stack_path,
        diff_output_path,
        baseline_output_path,
    ]
    for p in required:
        if not p.exists():
            print(f"[ERROR] Missing file: {p}")
            return

    # Load images
    img = load_image_2d(input_path)
    gt = load_image_2d(label_path)
    pred_stack = load_pred_stack(pred_stack_path)  # (N, H, W)
    diff_output = load_image_2d(diff_output_path)
    baseline_output = load_image_2d(baseline_output_path)

    n_samples = pred_stack.shape[0]
    n_use = min(max_samples, n_samples)

    print(f"[INFO] {dataset}/{image_id}: stack has {n_samples} slices, using {n_use}")

    # Final layout columns:
    # Input | GT | Sample1..SampleN | DiffusionOutput | BaselineOutput
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

    axes[col].imshow(diff_output, cmap="nipy_spectral")
    axes[col].set_title("Diffusion Out")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(baseline_output, cmap="nipy_spectral")
    axes[col].set_title("Baseline Out")
    axes[col].axis("off")

    fig.suptitle(f"{dataset} – ID {image_id}", y=0.97)
    plt.tight_layout()

    out_png = FIG_DIR / f"{dataset}_fullpanel_{image_id}.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"[SAVED] {out_png}")

def main():
    for dataset, cfg in CONFIG.items():
        build_panel(dataset, cfg, max_samples=5)

if __name__ == "__main__":
    main()
