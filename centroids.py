"""
Part C: Zentroide der Wirbelkoerper (ohne posteriore Elemente).

Berechnet auf Vorhersage-Masken (pred.nii.gz), nicht auf Ground-Truth.
Erst predict_and_save.py ausfuehren, dann dieses Skript.

Methode:
    Fuer jeden Wirbel wird entlang der Y-Achse (posterior -> anterior in RAS)
    pro Schicht der Kompaktheitsindex berechnet:
        compactness(y) = Flaeche(Maske in X-Z) / Flaeche(Bounding-Box in X-Z)
    Nach Glaettung mit einem Fenster von 5 Schichten wird die erste Schicht
    gesucht, ab der der Index dauerhaft ueber COMPACTNESS_THRESH (0.65) liegt.
    Alles anterior (hoehere Y-Werte in RAS) gilt als Wirbelkoerper.
    Der Zentroid ist der Mittelwert aller Wirbelkoerper-Voxel in Weltkoordinaten.

Weltkoordinaten:
    Die NIfTI-Affine in image.nii.gz enthaelt die echte Origin.
    Zentroid in Voxeln wird mit der Affine in mm umgerechnet.

Ausgabe:
    Ein JSON pro Fall (Aufgabenformat), plus centroids_all.csv als Uebersicht.

Nutzung:
    python centroids.py --pred-dir preds/totalseg --img-dir processed/totalseg --out-dir centroids/totalseg
    python centroids.py --pred-dir preds/verse    --img-dir processed/verse    --out-dir centroids/verse
    (--limit 5 fuer kurzen Test)
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

COMPACTNESS_THRESH = 0.4
SMOOTH_WINDOW      = 2
LUMBAR_NAMES       = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}


def compactness_profile(mask_2d_xz):
    if mask_2d_xz.sum() == 0:
        return 0.0
    xi, zi = np.where(mask_2d_xz)
    bbox_area = (xi.max() - xi.min() + 1) * (zi.max() - zi.min() + 1)
    return float(mask_2d_xz.sum()) / bbox_area


def find_body_cut(lab_k):
    """
    Geht von anterior (hohe Y-Werte) nach posterior (niedrige Y-Werte).
    Wartet bis der Index einmal ueber die Schwelle gestiegen ist (echter Koerper),
    dann schneidet er beim ersten dauerhaften Abfall darunter.
    Gibt den Y-Index zurueck, ab dem (inklusive) der Koerper liegt (alles >= cut behalten).
    """
    n_y = lab_k.shape[1]
    profile = np.array([compactness_profile(lab_k[:, y, :]) for y in range(n_y)])
    kernel  = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    smooth  = np.convolve(profile, kernel, mode="same")

    # Von anterior (n_y-1) nach posterior (0)
    seen_high = False
    for y in range(n_y - 1, -1, -1):
        if smooth[y] >= COMPACTNESS_THRESH:
            seen_high = True
        elif seen_high:
            # Wert war schon hoch (Koerper gesehen) und ist jetzt dauerhaft gefallen
            window_start = max(0, y - SMOOTH_WINDOW + 1)
            if np.mean(smooth[window_start:y + 1]) < COMPACTNESS_THRESH:
                return y + 1   # alles ab y+1 (anterior) ist Wirbelkoerper
    return 0   # kein Abfall gefunden: ganzen Wirbel als Koerper behandeln


def vertebra_body_mask(lab_k):
    cut  = find_body_cut(lab_k)
    body = lab_k.copy()
    if cut is not None and cut > 0:
        body[:, :cut, :] = False
    return body


def voxel_to_world(voxel_coords, affine):
    v = np.column_stack([voxel_coords, np.ones(len(voxel_coords))])
    return (affine @ v.T).T[:, :3]


def compute_centroid(pred_path, image_path):
    """Berechnet Zentroide pro Wirbel aus der Vorhersage-Maske (pred.nii.gz)."""
    affine = nib.load(image_path).affine          # echte Affine aus dem Bild
    pred   = np.asanyarray(nib.load(pred_path).dataobj).astype(np.uint8)

    centroids = {}
    for k, name in LUMBAR_NAMES.items():
        mask = pred == k
        if not mask.any():
            continue
        body = vertebra_body_mask(mask)
        if not body.any():
            body = mask                           # Fallback: ganzen Wirbel nehmen
        voxels       = np.argwhere(body).astype(float)
        centroid_vox = voxels.mean(axis=0)
        centroid_mm  = voxel_to_world(centroid_vox[None], affine)[0]
        centroids[name] = [round(float(v), 2) for v in centroid_mm]
    return centroids


def process(pred_dir, img_dir, out_dir, limit=None):
    pred_dir = Path(pred_dir)
    img_dir  = Path(img_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(d for d in pred_dir.iterdir()
                   if d.is_dir() and (d / "pred.nii.gz").exists())
    if limit:
        cases = cases[:limit]

    rows = []
    for d in cases:
        cid    = d.name
        pred_p = d / "pred.nii.gz"
        img_p  = img_dir / cid / "image.nii.gz"
        if not img_p.exists():
            print(f"  [warn] kein image fuer {cid}")
            continue
        centroids = compute_centroid(pred_p, img_p)
        result = {"case_id": cid, "coordinate_system": "RAS",
                  "unit": "mm", "centroids": centroids}
        (out_dir / f"{cid}_centroids.json").write_text(json.dumps(result, indent=2))
        row = {"case": cid}
        for name, coords in centroids.items():
            row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"] = coords
        rows.append(row)
        print(f"  {cid}: {list(centroids.keys())}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="Ordner mit <case>/pred.nii.gz")
    ap.add_argument("--img-dir",  required=True, help="Ordner mit <case>/image.nii.gz (fuer Affine)")
    ap.add_argument("--out-dir",  required=True, help="Ausgabe-Ordner fuer JSON + CSV")
    ap.add_argument("--limit",    type=int, default=None)
    a = ap.parse_args()

    rows = process(a.pred_dir, a.img_dir, a.out_dir, a.limit)
    df   = pd.DataFrame(rows)
    df.to_csv(Path(a.out_dir) / "centroids_all.csv", index=False)
    print(f"\n{len(df)} Faelle -> {a.out_dir}/centroids_all.csv")


if __name__ == "__main__":
    main()