"""
Part A: Preprocessing-Pipeline für VerSe + TotalSegmentator.

Ausgabe:
    processed/
        manifest.csv
        verse/<case>/image.nii.gz, label.nii.gz
        totalseg/<case>/image.nii.gz, label.nii.gz

Konventionen:
    Orientierung  : RAS (haeufigste Orientierung ueber beide Datensaetze)
    Spacing       : 2.0 mm isotrop (Empfehlung der Aufgabenstellung)
    dtype Bild    : float32, Labels uint8
    Labels        : 0=Hintergrund, 1..5 = L1..L5  (L6/T13 -> Hintergrund)
    Intensitaeten : Clip auf HU_WINDOW, dann linear auf [0,1]
    Crop          : Bounding-Box der Lendenwirbel + Rand (CROP_MARGIN_MM)

Nutzung:
    python preprocess.py
    # oder im Notebook:  from preprocess import *; df = run_all()
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom, label as cc

# ----------------------------------------------------------------- Konfiguration
BASE = Path(r"C:\Users\anmol\Documents\VSCode\application_BMDNow")
VERSE_ROOTS = {
    "verse19_train": BASE / "dataset-verse19training",
    "verse19_test":  BASE / "dataset-verse19test",
}
TOTALSEG_ROOT = BASE / "Totalsegmentator_dataset"
OUT_ROOT      = BASE / "processed"

TARGET_ORIENT  = "RAS"
TARGET_SPACING = (2.0, 2.0, 2.0)
HU_WINDOW      = (-500.0, 1300.0)   # unten Weichteil/Luftgrenze, oben kompakter Knochen
CROP_MARGIN_MM = 20.0

LUMBAR_NAMES = ["L1", "L2", "L3", "L4", "L5"]
VERSE_LUMBAR = {20: 1, 21: 2, 22: 3, 23: 4, 24: 5}   # VerSe-Label -> eigene Klasse
# --------------------------------------------------------------------------------


def to_orientation(img, target=TARGET_ORIENT):
    """Reorientiert ein NIfTI-Bild verlustfrei (nur Achsentausch/Flip)."""
    cur = nib.orientations.io_orientation(img.affine)
    tgt = nib.orientations.axcodes2ornt(tuple(target))
    return img.as_reoriented(nib.orientations.ornt_transform(cur, tgt))


def resample(arr, src_spacing, dst_spacing=TARGET_SPACING, is_label=False):
    """Bild: linear (order=1). Labels: Nearest-Neighbor (order=0), sonst
    entstehen Zwischenwerte, die keiner echten Klasse entsprechen."""
    factors = [s / d for s, d in zip(src_spacing, dst_spacing)]
    if np.allclose(factors, 1.0):
        return arr
    return zoom(arr, factors, order=0 if is_label else 1,
                mode="nearest", prefilter=not is_label)


def normalize(ct, window=HU_WINDOW):
    lo, hi = window
    return ((np.clip(ct, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def crop_to_labels(ct, lab, margin_mm=CROP_MARGIN_MM, spacing=TARGET_SPACING):
    """Bounding-Box der Lendenwirbel + Rand. Reduziert Volumen drastisch."""
    idx = np.argwhere(lab > 0)
    if idx.size == 0:
        return ct, lab, None
    lo = idx.min(0)
    hi = idx.max(0) + 1
    m  = np.array([int(round(margin_mm / s)) for s in spacing])
    lo = np.maximum(lo - m, 0)
    hi = np.minimum(hi + m, lab.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return ct[sl], lab[sl], (lo.tolist(), hi.tolist())


def save_case(out_dir, ct, lab, affine):
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(ct.astype(np.float32), affine), out_dir / "image.nii.gz")
    nib.save(nib.Nifti1Image(lab.astype(np.uint8),  affine), out_dir / "label.nii.gz")


# ------------------------------------------------------------------- Loader VerSe
def load_verse(ct_path, seg_path):
    ct_img  = to_orientation(nib.load(ct_path))
    seg_img = to_orientation(nib.load(seg_path))
    ct  = np.asanyarray(ct_img.dataobj).astype(np.float32)
    raw = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int32)
    lab = np.zeros_like(raw, dtype=np.uint8)
    for src, dst in VERSE_LUMBAR.items():
        lab[raw == src] = dst           # alles andere (inkl. L6=25, T13=28) bleibt 0
    return ct, lab, ct_img.header.get_zooms()[:3]


# ----------------------------------------------------------- Loader TotalSegmentator
def load_totalseg(case_dir):
    ct_img = to_orientation(nib.load(case_dir / "ct.nii.gz"))
    ct  = np.asanyarray(ct_img.dataobj).astype(np.float32)
    lab = np.zeros(ct.shape, dtype=np.uint8)
    for i, nm in enumerate(LUMBAR_NAMES, start=1):
        p = case_dir / "segmentations" / f"vertebrae_{nm}.nii.gz"
        if not p.exists():
            continue
        m = np.asanyarray(to_orientation(nib.load(p)).dataobj) > 0
        lab[m] = i                      # Binaermasken -> eine Multi-Label-Maske
    return ct, lab, ct_img.header.get_zooms()[:3]


# ------------------------------------------------------------------ Verarbeitung
def remove_small_fragments(lab, rel_thresh=0.05):
    """Pro Wirbel-Label: Komponenten < rel_thresh * groesste Komponente entfernen."""
    out = lab.copy()
    n_removed = 0
    for k in range(1, 6):
        comp, n = cc(lab == k)
        if n <= 1:
            continue
        sizes = np.bincount(comp.ravel())[1:]
        for i, s in enumerate(sizes, start=1):
            if s < rel_thresh * sizes.max():
                out[comp == i] = 0
                n_removed += int(s)
    return out, n_removed


def process(ct, lab, src_spacing, src_affine):
    """Resampling, Fragment-Bereinigung, Normalisierung, Crop.
    Gibt auch die echte NIfTI-Affine des gecroppten Volumens zurueck."""
    ct  = resample(ct,  src_spacing, is_label=False)
    lab = resample(lab, src_spacing, is_label=True)
    lab, n_removed = remove_small_fragments(lab)
    ct  = normalize(ct)
    ct, lab, bbox = crop_to_labels(ct, lab)

    # Weltkoordinaten des ersten Crop-Voxels (nach Resampling + Crop)
    lo     = np.array(bbox[0]) if bbox else np.zeros(3)
    origin = src_affine[:3, 3] + src_affine[:3, :3] @ (lo * np.array(TARGET_SPACING))
    crop_affine = np.diag(list(TARGET_SPACING) + [1.0]).astype(np.float64)
    crop_affine[:3, 3] = origin

    return ct, lab, bbox, n_removed, crop_affine


def run_all(out_root=OUT_ROOT):
    rows = []

    # --- VerSe
    for subset, root in VERSE_ROOTS.items():
        if not root.exists():
            print(f"[warn] fehlt: {root}")
            continue
        raw, der = root / "rawdata", root / "derivatives"
        for sub_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
            for ct_path in sorted(sub_dir.glob("*_ct.nii.gz")):
                cid      = ct_path.name[: -len("_ct.nii.gz")]
                seg_path = der / sub_dir.name / f"{cid}_seg-vert_msk.nii.gz"
                if not seg_path.exists():
                    continue
                src_img = nib.load(ct_path)
                ct, lab, sp = load_verse(ct_path, seg_path)
                n_lumbar = int(sum((lab == i).any() for i in range(1, 6)))
                if n_lumbar == 0:
                    rows.append(dict(dataset="verse", subset=subset, case=cid,
                                     subject=sub_dir.name, included=False,
                                     reason="kein L1-L5 im FOV", n_lumbar=0))
                    continue
                ct, lab, bbox, n_rm, crop_affine = process(ct, lab, sp, src_img.affine)
                save_case(out_root / "verse" / cid, ct, lab, crop_affine)
                rows.append(dict(
                    dataset="verse", subset=subset, case=cid, subject=sub_dir.name,
                    included=True, reason="-", n_lumbar=n_lumbar,
                    removed_fragment_vox=n_rm,
                    orig_orientation="".join(nib.aff2axcodes(src_img.affine)),
                    orig_spacing=tuple(round(float(z), 3) for z in sp),
                    orig_dtype=str(src_img.get_data_dtype()),
                    proc_shape=tuple(ct.shape), bbox=bbox,
                    origin_world=crop_affine[:3, 3].tolist(),
                    **{f"has_{n}": bool((lab == i).any())
                       for i, n in enumerate(LUMBAR_NAMES, start=1)},
                ))

    # --- TotalSegmentator
    if TOTALSEG_ROOT.exists():
        meta = None
        for cand in (TOTALSEG_ROOT / "meta.csv", TOTALSEG_ROOT.parent / "meta.csv"):
            if cand.exists():
                meta = pd.read_csv(cand, sep=";").set_index("image_id")
                break
        for case_dir in sorted(p for p in TOTALSEG_ROOT.iterdir() if p.is_dir()):
            if not (case_dir / "ct.nii.gz").exists():
                continue
            cid     = case_dir.name
            src_img = nib.load(case_dir / "ct.nii.gz")
            ct, lab, sp = load_totalseg(case_dir)
            n_lumbar = int(sum((lab == i).any() for i in range(1, 6)))
            if n_lumbar == 0:
                rows.append(dict(dataset="totalseg", subset="small", case=cid,
                                 subject=cid, included=False,
                                 reason="kein L1-L5 im FOV", n_lumbar=0))
                continue
            ct, lab, bbox, n_rm, crop_affine = process(ct, lab, sp, src_img.affine)
            save_case(out_root / "totalseg" / cid, ct, lab, crop_affine)
            row = dict(
                dataset="totalseg", subset="small", case=cid, subject=cid,
                included=True, reason="-", n_lumbar=n_lumbar,
                removed_fragment_vox=n_rm,
                orig_orientation="".join(nib.aff2axcodes(src_img.affine)),
                orig_spacing=tuple(round(float(z), 3) for z in sp),
                orig_dtype=str(src_img.get_data_dtype()),
                proc_shape=tuple(ct.shape), bbox=bbox,
                origin_world=crop_affine[:3, 3].tolist(),
                **{f"has_{n}": bool((lab == i).any())
                   for i, n in enumerate(LUMBAR_NAMES, start=1)},
            )
            if meta is not None and cid in meta.index:
                for col in ("manufacturer", "scanner_model", "kvp", "pathology_location"):
                    if col in meta.columns:
                        row[col] = meta.loc[cid, col]
            rows.append(row)

    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "manifest.csv", index=False)
    inc = df[df.included]
    print(f"{len(df)} Faelle gesichtet, {len(inc)} verarbeitet -> {out_root / 'manifest.csv'}")
    print(inc.groupby("dataset").size().to_string())
    if len(inc):
        shapes = np.array(inc["proc_shape"].dropna().tolist())
        print(f"Shapes nach Crop: min={shapes.min(0)}, median={np.median(shapes, 0)}, max={shapes.max(0)}")
    return df


if __name__ == "__main__":
    run_all()