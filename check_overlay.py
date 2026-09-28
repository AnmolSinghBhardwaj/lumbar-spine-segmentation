"""Sichtpruefung nach dem Preprocessing: Bild + Label ueberlagert.
Wenn die Maske nicht auf den Wirbeln liegt, stimmt Orientierung oder Resampling nicht.

Nutzung:  python check_overlay.py   (oder im Notebook: from check_overlay import show)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

OUT_ROOT = Path(r"C:\Users\anmol\Documents\VSCode\application_BMDNow\processed")


def show(case_dir):
    case_dir = Path(case_dir)
    ct = np.asanyarray(nib.load(case_dir / "image.nii.gz").dataobj)
    lab = np.asanyarray(nib.load(case_dir / "label.nii.gz").dataobj)

    # Sagittale Schicht durch den Schwerpunkt der Labels (dort sieht man alle Wirbel)
    idx = np.argwhere(lab > 0)
    x = int(idx[:, 0].mean()) if idx.size else ct.shape[0] // 2

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(ct[x].T, cmap="gray", origin="lower")
    ax[0].set_title(f"{case_dir.name} – CT")
    ax[1].imshow(ct[x].T, cmap="gray", origin="lower")
    ax[1].imshow(np.ma.masked_where(lab[x].T == 0, lab[x].T),
                 cmap="jet", alpha=0.5, origin="lower", vmin=1, vmax=5)
    ax[1].set_title("Overlay L1–L5")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()
    print(f"shape={ct.shape}  intensity=[{ct.min():.2f},{ct.max():.2f}]  labels={np.unique(lab)}")


if __name__ == "__main__":
    for ds in ("verse", "totalseg"):
        cases = sorted((OUT_ROOT / ds).iterdir())
        for c in cases[:2]:
            show(c)
