# Lumbar Spine Segmentation from CT (L1–L5)

A small end-to-end pipeline for lumbar spine CT: it harmonises two public datasets (TotalSegmentator and VerSe), trains a 3D U-Net to segment the vertebrae L1–L5, extracts the vertebral body centroids and evaluates how well the model transfers across datasets.

The model was trained on the TotalSegmentator small subset (93 cases after preprocessing) and evaluated in-domain on a held-out TotalSegmentator split and out-of-domain on VerSe'19.

<p align="center">
  <img src="bsp.png" alt="Example segmentation" width="700">
</p>
<p align="center">
  <img src="png.png" alt="Example segmentation" width="700">
</p>
<!-- Placeholder: add a sagittal slice with the L1–L5 prediction overlay here -->

---

## Quick start

```bash
./run_inference.sh <input_dir> <output_dir>
```

**Example:**
```bash
./run_inference.sh dataset-verse19test out/results
```

On Windows (CMD or PowerShell):
```bat
pip install -r requirements.txt
python inference.py --input dataset-verse19test --output out/results --weights best.pt
```

---

## Requirements

| Item | Value |
|---|---|
| OS | Linux, macOS, Windows (Git Bash for `.sh`) |
| Python | >= 3.10 |
| GPU | Not required (falls back to CPU automatically) |
| Docker | Not used |
| RAM | approx. 4 GB (CPU inference on cropped volumes) |

Dependencies are installed automatically on start (`pip install -r requirements.txt`). Main packages: `torch`, `monai`, `nibabel`, `scipy`, `numpy`, `pandas`.

---

## Input format

The script detects the VerSe and TotalSegmentator folder layouts automatically.

**VerSe:**
```
input_dir/
  rawdata/
    sub-verseXXX/
      sub-verseXXX_ct.nii.gz
  derivatives/
    sub-verseXXX/
      sub-verseXXX_seg-vert_msk.nii.gz   <- optional, used for Dice
```

**TotalSegmentator:**
```
input_dir/
  sXXXX/
    ct.nii.gz
    segmentations/
      vertebrae_L1.nii.gz                <- optional, used for Dice
      ...
```

Ground-truth labels are optional. If present, the volume is cropped to the lumbar region and Dice scores are computed.

---

## Output

One subfolder per case is created in `<output_dir>`:

```
output_dir/
  sub-verseXXX/
    pred.nii.gz          <- segmentation mask (0 = background, 1 = L1 ... 5 = L5)
    centroids.json       <- vertebral body centroids, RAS, mm
  dice.csv               <- Dice per vertebra (only if ground truth is available)
```

**Example `centroids.json`:**
```json
{
  "case_id": "sub-verse271",
  "coordinate_system": "RAS",
  "unit": "mm",
  "centroids": {
    "L1": [12.3, -45.6, 78.9],
    "L2": [12.1, -44.8, 49.2],
    "L3": [11.8, -44.1, 20.1],
    "L4": [11.5, -43.5, -9.4],
    "L5": [11.2, -42.8, -38.7]
  }
}
```

---

## Method

**Preprocessing.** Reorientation to RAS, resampling to 2 mm isotropic (image: linear, mask: nearest neighbour), fixed HU window [-500, 1300] scaled to [0, 1], crop to the lumbar bounding box plus a 20 mm margin. Label conventions of both datasets are mapped to 0 = background, 1–5 = L1–L5; transitional vertebrae (L6, T13) are set to background. Small stray label fragments (< 5 % of the largest component per vertebra) are removed.

**Model.** 3D U-Net (MONAI), 6 classes (background + L1–L5), channels 16–256, trained on 64×64×96 patches with mixed precision and a Dice + cross-entropy loss. AdamW (lr 1e-3, cosine decay), batch size 4, 6000 iterations, about 13 minutes on 2× NVIDIA T4 (Kaggle). No pretrained spine models were used.

**Vertebral body centroids.** For each predicted vertebra, a compactness profile (mask area / bounding-box area in the x–z plane) is computed slice by slice along the anterior–posterior axis and smoothed over 5 slices. Starting anteriorly, the body is cut off at the first sustained drop below 0.65 after the profile has exceeded it once. The vertebral body is compact and fills its bounding box, while the posterior arch does not. The centroid is the voxel mean of the body, converted to world coordinates.

---

## Results

Dice per vertebra (mean ± std), evaluated on ground-truth-cropped volumes:

| Test set | L1 | L2 | L3 | L4 | L5 | Mean |
|---|---|---|---|---|---|---|
| TotalSegmentator, own split (n = 15) | 0.861 ± 0.24 | 0.867 ± 0.24 | 0.900 ± 0.14 | 0.927 ± 0.04 | 0.899 ± 0.07 | 0.891 |
| VerSe'19 (n = 93–96) | 0.888 ± 0.10 | 0.882 ± 0.15 | 0.879 ± 0.17 | 0.859 ± 0.19 | 0.826 ± 0.20 | 0.853 |

Most of the variance comes from a few cases in which a vertebra lies partly outside the field of view: the model then shifts the level labels by one, which drives the Dice of individual vertebrae close to zero.

---

## Limitations

The model was trained on ground-truth-cropped volumes. Without labels, the full volume is used as input, which the model was not trained for, and performance can drop because there is no localisation stage. A two-stage pipeline (coarse localisation of the lumbar region, then fine segmentation) would be the next step towards a robust solution.

Further limitations:
- No sliding-window inference on raw whole-body CTs
- The TotalSegmentator test set is a custom split of the small subset, not the official test split
- Centroid world coordinates: the origin of reoriented volumes is not yet handled correctly for all input orientations; the quantitative comparison against the VerSe centroid annotations is still open, a qualitative comparison shows good visual agreement
- Single training run, single seed

---

## Notebooks

| Notebook | Content |
|---|---|
| `plots.ipynb` | Simple segmentation plots: CT slices with ground-truth and predicted L1–L5 overlays |
| `totalsegmentator_vs_verse_test_comparison.ipynb` | In-domain vs. out-of-domain comparison: Dice per vertebra, failure cases, centroid extraction and comparison with VerSe annotations |

---

## Files

| File | Purpose |
|---|---|
| `run_inference.sh` | Entry point (single command) |
| `inference.py` | Full inference pipeline |
| `train.py` | Model definition, training and helper functions (imported) |
| `preprocess.py` | Preprocessing functions (imported) |
| `centroids.py` | Vertebral body centroid extraction (imported) |
| `best.pt` | Trained model weights |
| `requirements.txt` | Python dependencies |
| `plots.ipynb` | Segmentation plots |
| `totalsegmentator_vs_verse_test_comparison.ipynb` | Cross-dataset analysis and centroids |

---

## Data

The datasets are not included in this repository and have to be downloaded separately:

- **TotalSegmentator** – Wasserthal et al., *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images*, Radiology: AI, 2023. Small subset v2.0.1, [Zenodo 10047263](https://zenodo.org/records/10047263) (CC BY 4.0).
- **VerSe** – Sekuboyina et al., *VerSe: A Vertebrae Labelling and Segmentation Benchmark for Multi-detector CT Images*, Medical Image Analysis, 2021; Löffler et al., Radiology: AI, 2020; Liebl et al., Scientific Data, 2021. [github.com/anjany/verse](https://github.com/anjany/verse) (CC BY-SA 4.0).

---

## AI usage

AI assistants (Claude, Anthropic) were used as a coding assistant and for drafting text. All design decisions (orientation, spacing, HU window, model architecture, centroid method) were made and justified by the author.
