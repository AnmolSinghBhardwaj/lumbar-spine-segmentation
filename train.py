
"""
Part B: 3D-U-Net-Training auf den vorverarbeiteten TotalSegmentator-Crops.

Laeuft auf Kaggle, Colab, Uni-Cluster und (zum Testen) auf der CPU.

Kaggle-Nutzung (3 Zellen):
    !pip install -q monai
    !python train.py --smoke               # kurzer Funktionstest
    !python train.py --minutes 60          # richtiges Training

Erwartete Daten: irgendwo unter --data-root liegen Ordner  .../totalseg/<case>/image.nii.gz
und label.nii.gz (Ausgabe von preprocess.py). Der Pfad wird automatisch gesucht.

Ausgaben (in --out-dir, auf Kaggle /kaggle/working):
    best.pt, last.pt   Gewichte (best = beste Validierungs-Dice)
    split.json         welche Faelle in Train / Val / Test liegen
    log.csv            Verlauf
    test_dice.csv      Dice pro Wirbel und Fall auf dem eigenen Test-Split

Modell: 3D-U-Net (MONAI), 6 Klassen (0 = Hintergrund, 1..5 = L1..L5), Training auf Patches,
Mixed Precision, Loss = Dice + Cross-Entropy. Nichts Vortrainiertes.
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet

NUM_CLASSES = 6                      # 0 = Hintergrund, 1..5 = L1..L5
PATCH = (64, 64, 96)                 # Trainings-Ausschnitt in Voxeln (bei 2 mm: 128 x 128 x 192 mm)
CHANNELS = (16, 32, 64, 128, 256)
STRIDES = (2, 2, 2, 2)
EXCLUDE = {"s0573", "s1221"}         # in der Sanity-Pruefung aufgefallen (Wirbelreihenfolge)


# ----------------------------------------------------------------------------- Daten
def _pick(d, stem):
    """Findet <stem>.nii.gz oder <stem>.nii (Kaggle entpackt .gz beim Upload)."""
    for ext in (".nii.gz", ".nii"):
        if (d / (stem + ext)).exists():
            return d / (stem + ext)
    return None


def find_case_dirs(root):
    dirs, seen = [], set()
    root = Path(root)
    for p in sorted(list(root.rglob("image.nii.gz")) + list(root.rglob("image.nii"))):
        d = p.parent
        if d in seen:
            continue
        seen.add(d)
        if _pick(d, "label") is not None and d.name not in EXCLUDE and d.parent.name != "verse":
            dirs.append(d)
    return dirs


def load_case(d):
    img = nib.load(_pick(d, "image")).get_fdata(dtype=np.float32)
    lab = np.asanyarray(nib.load(_pick(d, "label")).dataobj).astype(np.uint8)
    return img, lab


def make_split(cases, n_test=15, n_val=8, seed=0):
    """Eigener Split. Test nur aus Faellen mit vollstaendigem L1-L5 (vergleichbare Dice pro Wirbel)."""
    rng = random.Random(seed)
    complete = [c for c in cases if all((cases[c][1] == k).any() for k in range(1, 6))]
    test = rng.sample(complete, min(n_test, len(complete)))
    rest = [c for c in cases if c not in test]
    rng.shuffle(rest)
    val = rest[:n_val]
    train = rest[n_val:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def pad_to(arr, size):
    pads = [(0, max(0, s - a)) for a, s in zip(arr.shape, size)]
    return np.pad(arr, pads) if any(p[1] for p in pads) else arr


def random_patch(img, lab):
    img, lab = pad_to(img, PATCH), pad_to(lab, PATCH)
    starts = [random.randint(0, a - p) for a, p in zip(img.shape, PATCH)]
    sl = tuple(slice(s, s + p) for s, p in zip(starts, PATCH))
    im, lb = img[sl], lab[sl]
    if random.random() < 0.5:                    # Links-Rechts-Spiegelung (Achse 0 in RAS)
        im, lb = im[::-1], lb[::-1]              # Kopf-Fuss-Spiegelung waere falsch (Wirbelreihenfolge)
    im = np.clip(im * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05), 0, 1)
    return np.ascontiguousarray(im, dtype=np.float32), np.ascontiguousarray(lb)


def make_batch(train_ids, cases, batch_size):
    ims, lbs = [], []
    for c in random.choices(train_ids, k=batch_size):
        im, lb = random_patch(*cases[c])
        ims.append(im)
        lbs.append(lb)
    x = torch.from_numpy(np.stack(ims))[:, None]
    y = torch.from_numpy(np.stack(lbs).astype(np.int64))[:, None]
    return x, y


# ---------------------------------------------------------------------------- Modell
def build_model():
    return UNet(spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES,
                channels=CHANNELS, strides=STRIDES, num_res_units=2)


@torch.no_grad()
def predict_volume(model, img, device, overlap=0.5):
    """Schiebefenster ueber das ganze (gecroppte) Volumen, gibt Label-Volumen zurueck."""
    model.eval()
    x = torch.from_numpy(img)[None, None].to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = sliding_window_inference(x, PATCH, sw_batch_size=2, predictor=model, overlap=overlap)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def dice_per_class(pred, gt):
    """Dice = 2*|Vorhersage & Wahrheit| / (|Vorhersage| + |Wahrheit|); NaN, wenn Wirbel in gt fehlt."""
    out = []
    for k in range(1, 6):
        g, p = gt == k, pred == k
        out.append(np.nan if g.sum() == 0 else 2.0 * (p & g).sum() / (p.sum() + g.sum()))
    return out


def save_ckpt(path, model):
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({"state_dict": m.state_dict(), "channels": CHANNELS, "strides": STRIDES,
                "patch": PATCH, "num_classes": NUM_CLASSES}, path)


def evaluate(model, ids, cases, device):
    rows = [dice_per_class(predict_volume(model, cases[c][0], device), cases[c][1]) for c in ids]
    return np.array(rows, dtype=float)


# --------------------------------------------------------------------------- Training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/kaggle/input" if Path("/kaggle/input").exists() else ".")
    ap.add_argument("--out-dir", default="/kaggle/working" if Path("/kaggle/working").exists() else "out")
    ap.add_argument("--minutes", type=float, default=60, help="Zeitbudget fuer das Training")
    ap.add_argument("--iters", type=int, default=4000, help="maximale Iterationen")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="kurzer Funktionstest")
    a = ap.parse_args()
    if a.smoke:
        a.minutes, a.iters, a.batch, a.val_every = 2, 4, 1, 2

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| GPUs:", torch.cuda.device_count())

    dirs = find_case_dirs(a.data_root)
    if a.smoke:
        dirs = dirs[:12]
    print(f"{len(dirs)} Faelle gefunden unter {a.data_root}")
    if not dirs:
        raise SystemExit("Keine Faelle gefunden - --data-root pruefen (Ordner mit .../<case>/image.nii.gz).")
    cases = {d.name: load_case(d) for d in dirs}

    split = make_split(cases, n_test=3 if a.smoke else 15, n_val=2 if a.smoke else 8, seed=a.seed)
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print({k: len(v) for k, v in split.items()})

    model = build_model().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    budget, t0, best = a.minutes * 60, time.time(), -1.0
    log = open(out / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["iter", "minutes", "loss", "val_mean_dice", "L1", "L2", "L3", "L4", "L5"])

    run_loss, it = 0.0, 0
    while it < a.iters and time.time() - t0 < budget:
        progress = max(it / a.iters, (time.time() - t0) / budget)     # Lernrate faellt nach Zeit ODER Iteration
        for g in opt.param_groups:
            g["lr"] = a.lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
        model.train()
        x, y = make_batch(split["train"], cases, a.batch)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        run_loss += loss.item()
        it += 1

        if it % a.val_every == 0 or it == a.iters:
            d = evaluate(model, split["val"], cases, device)
            per = np.nanmean(d, axis=0)
            mean = float(np.nanmean(per))
            w.writerow([it, round((time.time() - t0) / 60, 1), round(run_loss / a.val_every, 4), round(mean, 4)]
                       + [round(float(v), 4) for v in per]); log.flush()
            print(f"it {it:5d} | {(time.time() - t0) / 60:5.1f} min | loss {run_loss / a.val_every:.4f} "
                  f"| val Dice {mean:.3f} | pro Wirbel {np.round(per, 3)}")
            run_loss = 0.0
            save_ckpt(out / "last.pt", model)
            if mean > best:
                best = mean
                save_ckpt(out / "best.pt", model)
    log.close()
    if not (out / "best.pt").exists():
        save_ckpt(out / "best.pt", model)

    # ------------------------------------------------ Test auf eigenem Split (in-domain)
    ck = torch.load(out / "best.pt", map_location=device)
    m = build_model().to(device)
    m.load_state_dict(ck["state_dict"])
    d = evaluate(m, split["test"], cases, device)
    with open(out / "test_dice.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["case", "L1", "L2", "L3", "L4", "L5"])
        for c, row in zip(split["test"], d):
            wr.writerow([c] + [round(float(v), 4) for v in row])
    print("\nTEST (in-domain, eigener Split, Oracle-Crop) - Dice pro Wirbel, Mittel +- Std:")
    for k, (mu, sd) in enumerate(zip(np.nanmean(d, 0), np.nanstd(d, 0)), start=1):
        print(f"  L{k}: {mu:.3f} +- {sd:.3f}")
    print(f"Fertig nach {(time.time() - t0) / 60:.1f} min. Dateien in {out}")


if __name__ == "__main__":
    main()
