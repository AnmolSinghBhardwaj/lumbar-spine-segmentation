"""
Speichert Vorhersage-Masken (pred.nii.gz) fuer alle Faelle eines Datensatzes.
Wird vor centroids.py ausgefuehrt, damit Zentroide auf Predictions berechnet werden.

Nutzung:
    python predict_and_save.py --weights best.pt --data processed/totalseg --out preds/totalseg
    python predict_and_save.py --weights best.pt --data processed/verse    --out preds/verse
    (--limit 5 fuer kurzen Test)
"""
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import processed.train as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data",    required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--limit",   type=int, default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ck = torch.load(a.weights, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(a.weights, map_location=device)
    model = T.build_model().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    dirs = [d for d in sorted(Path(a.data).iterdir())
            if d.is_dir() and T._pick(d, "image")]
    if a.limit:
        dirs = dirs[:a.limit]
    print(f"{len(dirs)} Faelle | device={device}")

    out_root = Path(a.out)
    for i, d in enumerate(dirs, 1):
        img_p = T._pick(d, "image")
        img_nib = nib.load(img_p)                        # echte Affine erhalten
        img_arr = img_nib.get_fdata(dtype=np.float32)
        pred = T.predict_volume(model, img_arr, device)  # uint8 Array

        out_dir = out_root / d.name
        out_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred, img_nib.affine), out_dir / "pred.nii.gz")
        if i % 10 == 0 or i == len(dirs):
            print(f"  {i}/{len(dirs)}")

    print(f"Fertig -> {out_root}")


if __name__ == "__main__":
    main()
