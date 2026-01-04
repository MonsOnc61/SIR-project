import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path


def plot_slices_with_overlay(volume, mask=None, output_png="output.png"):
    """
    Crée une image composée des coupes axiale, coronale et sagittale d'un volume.
    Si un masque multi-label est fourni, il est superposé en couleur.
    """

    # ===== Conversion volume =====
    vol_np = sitk.GetArrayFromImage(volume)  # (z, y, x)
    zc, yc, xc = (s // 2 for s in vol_np.shape)

    axial = vol_np[zc, :, :]
    coronal = vol_np[:, yc, :]
    sagittal = vol_np[:, :, xc]

    # ===== Conversion masque =====
    if mask is not None:
        mask_np = sitk.GetArrayFromImage(mask)

        mask_axial = mask_np[zc, :, :]
        mask_coronal = mask_np[:, yc, :]
        mask_sagittal = mask_np[:, :, xc]

        # Labels réels (hors background)
        labels = np.unique(mask_np)
        labels = labels[labels != 0]
        n_labels = len(labels)

        # Colormap discrète
        base_colors = plt.cm.tab20(np.linspace(0, 1, 50))
        cmap = ListedColormap(base_colors[:n_labels])

        # Norm basé UNIQUEMENT sur indices remappés 1..n_labels
        norm = BoundaryNorm(
            boundaries=np.arange(0.5, n_labels + 1.5, 1),
            ncolors=n_labels
        )

    else:
        mask_axial = mask_coronal = mask_sagittal = None
        mask_np = None
        cmap = norm = labels = None

    # ===== Plot =====
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    slices = [
        (axial, mask_axial, "Axial"),
        (coronal, mask_coronal, "Coronal"),
        (sagittal, mask_sagittal, "Sagittal"),
    ]

    for ax, (slc, mask_slc, title) in zip(axes, slices):
        ax.imshow(slc.T, cmap="gray", origin="lower")

        if mask_slc is not None and labels.size > 0:
            # Remapping local des labels
            mapped = np.zeros_like(mask_slc, dtype=np.int32)
            for idx, label in enumerate(labels, start=1):
                mapped[mask_slc == label] = idx

            masked = np.ma.masked_where(mapped == 0, mapped)

            ax.imshow(
                masked.T,
                cmap=cmap,
                norm=norm,
                alpha=0.4,
                origin="lower",
                interpolation="none",
            )

        ax.set_title(title)
        ax.axis("off")

    # ===== Infos =====
    vol_min, vol_max = float(vol_np.min()), float(vol_np.max())
    vol_type = volume.GetPixelIDTypeAsString()

    if mask_np is not None:
        mask_min, mask_max = float(mask_np.min()), float(mask_np.max())
        mask_type = mask.GetPixelIDTypeAsString()
        label_info = f"{len(labels)} labels"
    else:
        mask_min = mask_max = "NA"
        mask_type = "None"
        label_info = "None"

    info_text = (
        f"Volume: min={vol_min:.2f}, max={vol_max:.2f}, type={vol_type}\n"
        f"Masque: min={mask_min}, max={mask_max}, type={mask_type}, labels={label_info}"
    )

    plt.figtext(0.5, -0.05, info_text, ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualisation orthogonale d'un volume NIfTI avec overlay masque multi-label"
    )
    parser.add_argument("--input", required=True, help="Volume NIfTI (.nii.gz)")
    parser.add_argument("--mask", help="Masque NIfTI optionnel (.nii.gz)")
    parser.add_argument("--output_png", default="output.png", help="PNG de sortie")
    parser.add_argument("--dest", default=str(Path.cwd()), help="Repertoire de destination")

    args = parser.parse_args()

    volume = sitk.ReadImage(args.input)
    mask = sitk.ReadImage(args.mask) if args.mask else None

    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)  # create dest directory if not exists
    output_path = dest_path / args.output_png

    plot_slices_with_overlay(volume, mask, output_path)


if __name__ == "__main__":
    main()