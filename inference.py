"""
Inferenz-Pipeline: Preprocessing -> Vorhersage -> Zentroide.
Wird von run_inference.sh aufgerufen.

Erkennt automatisch VerSe- und TotalSegmentator-Format.
Falls keine Ground-Truth-Labels vorhanden: kein Crop, ganzes Volumen.
Falls Labels vorhanden: Crop auf Lendenregion + Dice-Berechnung.

Ausgabe pro Fall in <output_dir>/<case_id>/:
    pred.nii.gz              Segmentierungsmaske (0=bg, 1=L1 .. 5=L5)
    centroids.json           Wirbelkoerper-Zentroide in RAS mm
Ausgabe gesamt:
    dice.csv                 Dice pro Wirbel (nur wenn GT vorhanden)
"""
import argparse, json, csv
from pathlib import Path
import nibabel as nib
import numpy as np
import torch

# Eigene Module (muessen im selben Ordner liegen)
import preprocess as P
import centroids  as C
import train      as T

WEIGHTS_DEFAULT = Path(__file__).parent / "best.pt"
LUMBAR_NAMES    = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}


# ---------------------------------------------------------------- Format-Erkennung
def detect_format(case_dir):
    """Gibt 'verse' oder 'totalseg' zurueck."""
    if (case_dir / "ct.nii.gz").exists():
        return "totalseg"
    for f in case_dir.glob("*_ct.nii.gz"):
        return "verse"
    return None


def find_cases(input_dir):
    """Sucht alle Faelle im Eingabe-Ordner."""
    input_dir = Path(input_dir)
    cases = []
    # VerSe: rawdata/sub-XXX/
    raw = input_dir / "rawdata"
    if raw.exists():
        for sub in sorted(raw.iterdir()):
            for ct in sorted(sub.glob("*_ct.nii.gz")):
                cid = ct.name[:-len("_ct.nii.gz")]
                der = input_dir / "derivatives" / sub.name
                seg = der / f"{cid}_seg-vert_msk.nii.gz"
                cases.append(("verse", cid, ct, seg if seg.exists() else None))
        return cases
    # TotalSegmentator: sXXXX/ct.nii.gz
    for sub in sorted(input_dir.iterdir()):
        ct = sub / "ct.nii.gz"
        if ct.exists():
            seg_dir = sub / "segmentations"
            cases.append(("totalseg", sub.name, sub, seg_dir if seg_dir.exists() else None))
    return cases


# ---------------------------------------------------------------- Laden + Preprocessing
def load_and_preprocess(fmt, cid, ct_path_or_dir, seg_path_or_dir):
    """Gibt (ct_arr, lab_arr_or_None, src_affine, src_spacing) zurueck."""
    if fmt == "verse":
        ct_img = P.to_orientation(nib.load(ct_path_or_dir))
        ct  = np.asanyarray(ct_img.dataobj).astype(np.float32)
        sp  = ct_img.header.get_zooms()[:3]
        aff = ct_img.affine
        lab = None
        if seg_path_or_dir is not None and seg_path_or_dir.exists():
            seg_img = P.to_orientation(nib.load(seg_path_or_dir))
            raw = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int32)
            lab = np.zeros_like(raw, dtype=np.uint8)
            for src, dst in P.VERSE_LUMBAR.items():
                lab[raw == src] = dst
    else:
        ct_img = P.to_orientation(nib.load(ct_path_or_dir / "ct.nii.gz"))
        ct  = np.asanyarray(ct_img.dataobj).astype(np.float32)
        sp  = ct_img.header.get_zooms()[:3]
        aff = ct_img.affine
        lab = None
        if seg_path_or_dir is not None and seg_path_or_dir.exists():
            lab = np.zeros(ct.shape, dtype=np.uint8)
            for i, nm in enumerate(P.LUMBAR_NAMES, start=1):
                p = seg_path_or_dir / f"vertebrae_{nm}.nii.gz"
                if p.exists():
                    m = np.asanyarray(P.to_orientation(nib.load(p)).dataobj) > 0
                    lab[m] = i

    # Resampling
    ct  = P.resample(ct,  sp, is_label=False)
    if lab is not None:
        lab = P.resample(lab, sp, is_label=True)
        lab, _ = P.remove_small_fragments(lab)

    ct = P.normalize(ct)

    # Crop: mit GT auf Lendenregion, ohne GT ganzes Volumen
    if lab is not None and lab.any():
        ct, lab, bbox = P.crop_to_labels(ct, lab)
        lo = np.array(bbox[0])
    else:
        bbox = None
        lo   = np.zeros(3)

    # Affine des verarbeiteten Volumens
    origin      = aff[:3, 3] + aff[:3, :3] @ (lo * np.array(P.TARGET_SPACING))
    crop_affine = np.diag(list(P.TARGET_SPACING) + [1.0]).astype(np.float64)
    crop_affine[:3, 3] = origin

    return ct, lab, crop_affine


# ---------------------------------------------------------------- Zentroide
def compute_centroids(pred, affine):
    centroids = {}
    for k, name in LUMBAR_NAMES.items():
        mask = pred == k
        if not mask.any():
            continue
        body = C.vertebra_body_mask(mask)
        if not body.any():
            body = mask
        vox = np.argwhere(body).astype(float).mean(0)
        mm  = (affine @ np.append(vox, 1))[:3]
        centroids[name] = [round(float(v), 2) for v in mm]
    return centroids


# ---------------------------------------------------------------- Dice
def dice_per_class(pred, gt):
    out = {}
    for k, name in LUMBAR_NAMES.items():
        g, p = gt == k, pred == k
        if g.sum() == 0:
            continue
        out[name] = round(2.0 * float((p & g).sum()) / float(p.sum() + g.sum()), 4)
    return out


# ---------------------------------------------------------------- Hauptfunktion
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   required=True)
    ap.add_argument("--output",  required=True)
    ap.add_argument("--weights", default=str(WEIGHTS_DEFAULT))
    ap.add_argument("--limit",   type=int, default=None)
    a = ap.parse_args()

    out_root = Path(a.output)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    try:
        ck = torch.load(a.weights, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(a.weights, map_location=device)
    model = T.build_model().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"Gewichte geladen: {a.weights}")

    cases = find_cases(a.input)
    if a.limit:
        cases = cases[:a.limit]
    print(f"{len(cases)} Faelle gefunden in {a.input}")

    dice_rows = []
    for i, (fmt, cid, ct_ref, seg_ref) in enumerate(cases, 1):
        print(f"  [{i}/{len(cases)}] {cid} ({fmt})")
        try:
            ct, lab, affine = load_and_preprocess(fmt, cid, ct_ref, seg_ref)
        except Exception as e:
            print(f"    [warn] Preprocessing fehlgeschlagen: {e}"); continue

        pred = T.predict_volume(model, ct, device)

        out_dir = out_root / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(pred, affine), out_dir / "pred.nii.gz")

        cents = compute_centroids(pred, affine)
        result = {"case_id": cid, "coordinate_system": "RAS",
                  "unit": "mm", "centroids": cents}
        (out_dir / "centroids.json").write_text(json.dumps(result, indent=2))

        if lab is not None:
            dc = dice_per_class(pred, lab)
            dc["case"] = cid
            dice_rows.append(dc)
            mean = np.mean(list(dc[v] for v in dc if v != "case"))
            print(f"    Dice mean={mean:.3f}  Wirbel={list(cents.keys())}")
        else:
            print(f"    (kein GT)  Wirbel={list(cents.keys())}")

    if dice_rows:
        keys = ["case"] + list(LUMBAR_NAMES.values())
        with open(out_root / "dice.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(dice_rows)
        vals = [[r[v] for v in LUMBAR_NAMES.values() if v in r] for r in dice_rows]
        flat = [v for row in vals for v in row]
        print(f"\nDice gesamt: {np.mean(flat):.3f} (n={len(dice_rows)} Faelle)")
        print(f"Ergebnisse -> {out_root}")


if __name__ == "__main__":
    main()
