import SimpleITK as sitk
import numpy as np
import math
import csv
import os
import imageio.v2 as imageio

# =========================
# Réglages
# =========================
MASK_PATH = "MSLesSeg_Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
OUT_CSV = "Output/P1_lesions_features_with_tchebichef.csv"
OUT_RECON_DIR = "Output/reconstructions"

FOURIER_K = 12
LEGENDRE_N = 3
CHEBYSHEV_N = 3
TCHEB_N = 6        # ordre Tchebichef (clé pour reconstruction)

CSV_DELIM = ";"

os.makedirs(OUT_RECON_DIR, exist_ok=True)

# =========================
# Utils bbox
# =========================
def bbox_from_mask(mask2d):
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max()-xs.min()+1, ys.max()-ys.min()+1

# =========================
# Polynômes Tchebichef discrets
# =========================
def tchebichef_poly(p, x, N):
    if p == 0:
        return np.ones_like(x, dtype=np.float64)
    if p == 1:
        return (2*x + 1 - N) / N

    t0 = np.ones_like(x, dtype=np.float64)
    t1 = (2*x + 1 - N) / N
    for k in range(1, p):
        a = (2*(2*k+1)*(2*x+1-N)) / ((k+1)*(N*N-(k+1)*(k+1)))
        b = (k*(N*N-k*k)) / ((k+1)*(N*N-(k+1)*(k+1)))
        t2 = a*t1 - b*t0
        t0, t1 = t1, t2
    return t1

# =========================
# Moments de Tchebichef 2D
# =========================
def tchebichef_moments(mask2d, N=6):
    bb = bbox_from_mask(mask2d)
    if bb is None:
        return [0.0]*(N*N), None

    x0,y0,w,h = bb
    roi = mask2d[y0:y0+h, x0:x0+w].astype(np.float64)

    xs = np.arange(w)
    ys = np.arange(h)

    feats = []
    for p in range(N):
        tp = tchebichef_poly(p, xs, w)
        for q in range(N):
            tq = tchebichef_poly(q, ys, h)
            feats.append(float(np.sum(roi * tq[:,None] * tp[None,:])))

    denom = max(1.0, np.sum(roi))
    feats = [v/denom for v in feats]
    return feats, bb

# =========================
# Reconstruction Tchebichef
# =========================
def reconstruct_from_tchebichef(moments, w, h, N):
    recon = np.zeros((h, w), dtype=np.float64)
    idx = 0
    xs = np.arange(w)
    ys = np.arange(h)

    for p in range(N):
        tp = tchebichef_poly(p, xs, w)
        for q in range(N):
            tq = tchebichef_poly(q, ys, h)
            recon += moments[idx] * tq[:,None] * tp[None,:]
            idx += 1

    recon -= recon.min()
    if recon.max() > 1e-12:
        recon /= recon.max()
    return recon

# =========================
# Distance de forme (article)
# =========================
def shape_distance(m, ref, sigma=2.0):
    m = np.array(m)
    ref = np.array(ref)
    P = int(np.sqrt(len(m)))

    weights = []
    for p in range(P):
        for q in range(P):
            weights.append(np.exp(-((p+q)**2)/(2*sigma*sigma)))
    weights = np.array(weights)

    return float(np.sum(weights * (m-ref)**2))

# =========================
# Chargement masque
# =========================
mask_img = sitk.ReadImage(MASK_PATH)
mask_bin = sitk.Cast(sitk.BinaryThreshold(mask_img, 0.5, 1e9, 1, 0), sitk.sitkUInt8)
mask_arr = sitk.GetArrayFromImage(mask_bin)
spacing = mask_bin.GetSpacing()
sx, sy, sz = spacing

# =========================
# CC 3D
# =========================
cc = sitk.ConnectedComponent(mask_bin)
cc_arr = sitk.GetArrayFromImage(cc)

labels = np.unique(cc_arr)
labels = labels[labels != 0]

# =========================
# Référence de forme (moyenne)
# =========================
all_tch = []

rows = []

for lab in labels:
    lesion = (cc_arr == lab).astype(np.uint8)

    best_z = np.argmax([np.sum(lesion[z]) for z in range(lesion.shape[0])])
    mask2d = lesion[best_z]

    tch, bb = tchebichef_moments(mask2d, TCHEB_N)
    if bb is None:
        continue

    all_tch.append(tch)

# moyenne = forme de référence
tch_ref = np.mean(np.array(all_tch), axis=0)

# =========================
# Extraction finale
# =========================
lesion_id = 1
for lab in labels:
    lesion = (cc_arr == lab).astype(np.uint8)
    best_z = np.argmax([np.sum(lesion[z]) for z in range(lesion.shape[0])])
    mask2d = lesion[best_z]

    tch, bb = tchebichef_moments(mask2d, TCHEB_N)
    if bb is None:
        continue

    x0,y0,w,h = bb
    recon = reconstruct_from_tchebichef(tch, w, h, TCHEB_N)

    recon_full = np.zeros_like(mask2d, dtype=np.float64)
    recon_full[y0:y0+h, x0:x0+w] = recon
    recon_bin = (recon_full > 0.5).astype(np.uint8)

    # sauvegarde image
    imageio.imwrite(
        os.path.join(OUT_RECON_DIR, f"lesion_{lesion_id}_slice_{best_z}.png"),
        (recon_full*255).astype(np.uint8)
    )

    # distance de forme
    dist = shape_distance(tch, tch_ref)

    rows.append({
        "lesion_id": lesion_id,
        "slice_z": int(best_z),
        "shape_distance": dist,
        **{f"tch_{i}": tch[i] for i in range(len(tch))}
    })

    lesion_id += 1

# =========================
# CSV
# =========================
fieldnames = ["lesion_id","slice_z","shape_distance"] + \
             [f"tch_{i}" for i in range(TCHEB_N*TCHEB_N)]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=CSV_DELIM)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"OK – {len(rows)} lésions traitées")
print(f"Reconstructions dans {OUT_RECON_DIR}")
print(f"CSV écrit : {OUT_CSV}")
