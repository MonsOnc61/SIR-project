import SimpleITK as sitk
import numpy as np
import os
import csv
from scipy.ndimage import zoom
import imageio.v2 as imageio

# ============================================================
# PARAMÈTRES
# ============================================================
MASK_PATH = "MSLesSeg_Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
OUT_CSV = "Output/P1_lesions_features_tchebichef.csv"
OUT_RECON_DIR = "Output/reconstructions"

ROI_SIZE = 64          # taille canonique (invariance échelle)
TCHEB_N = 20           # ordre moments
SIGMA_DISTANCE = 3.0
CSV_DELIM = ";"

os.makedirs(OUT_RECON_DIR, exist_ok=True)

# ============================================================
# UTILS
# ============================================================
def bbox_from_mask(mask2d):
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1

def resize_roi(mask, size):
    h, w = mask.shape
    zoom_y = size / h
    zoom_x = size / w
    return zoom(mask.astype(np.float64),
                (zoom_y, zoom_x),
                order=0)   # nearest-neighbor

# ============================================================
# POLYNÔMES DE TCHEBICHEF DISCRETS
# ============================================================
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

def tchebichef_norm(p, N):
    x = np.arange(N)
    tp = tchebichef_poly(p, x, N)
    return np.sum(tp * tp)

# ============================================================
# MOMENTS DE TCHEBICHEF 2D (CORRECTS)
# ============================================================
def tchebichef_moments(mask2d, N, roi_size):
    bb = bbox_from_mask(mask2d)
    if bb is None:
        return None, None

    x0, y0, w, h = bb
    roi = mask2d[y0:y0+h, x0:x0+w].astype(np.float64)

    roi = resize_roi(roi, roi_size)

    xs = np.arange(roi_size)
    ys = np.arange(roi_size)

    moments = np.zeros((N, N), dtype=np.float64)

    for p in range(N):
        tp = tchebichef_poly(p, xs, roi_size)
        np_p = tchebichef_norm(p, roi_size)

        for q in range(N):
            tq = tchebichef_poly(q, ys, roi_size)
            np_q = tchebichef_norm(q, roi_size)

            moments[p, q] = np.sum(roi * tq[:, None] * tp[None, :]) / (np_p * np_q)

    return moments.flatten(), bb

# ============================================================
# RECONSTRUCTION
# ============================================================
def reconstruct_from_tchebichef(moments, size, N):
    recon = np.zeros((size, size), dtype=np.float64)
    idx = 0
    xs = np.arange(size)
    ys = np.arange(size)

    for p in range(N):
        tp = tchebichef_poly(p, xs, size)
        for q in range(N):
            tq = tchebichef_poly(q, ys, size)
            recon += moments[idx] * tq[:, None] * tp[None, :]
            idx += 1

    recon -= recon.min()
    recon /= recon.max() + 1e-12
    return recon

def binarize_reconstruction(recon):
    vals = recon[recon > 0]
    if len(vals) == 0:
        return np.zeros_like(recon, dtype=np.uint8)
    th = np.percentile(vals, 70)
    return (recon > th).astype(np.uint8)

# ============================================================
# DISTANCE DE FORME
# ============================================================
def shape_distance(m, ref, sigma):
    m = np.array(m)
    ref = np.array(ref)
    P = int(np.sqrt(len(m)))

    d = 0.0
    idx = 0
    for p in range(P):
        for q in range(P):
            w = np.exp(-(p*p + q*q) / (2*sigma*sigma))
            d += w * (m[idx] - ref[idx])**2
            idx += 1
    return float(d)

# ============================================================
# CHARGEMENT MASQUE
# ============================================================
mask_img = sitk.ReadImage(MASK_PATH)
mask_bin = sitk.Cast(sitk.BinaryThreshold(mask_img, 0.5, 1e9, 1, 0),
                     sitk.sitkUInt8)
mask_arr = sitk.GetArrayFromImage(mask_bin)

cc = sitk.ConnectedComponent(mask_bin)
cc_arr = sitk.GetArrayFromImage(cc)

labels = np.unique(cc_arr)
labels = labels[labels != 0]

# ============================================================
# RÉFÉRENCE DE FORME (MOYENNE)
# ============================================================
all_moments = []

for lab in labels:
    lesion = (cc_arr == lab).astype(np.uint8)
    best_z = np.argmax([np.sum(lesion[z]) for z in range(lesion.shape[0])])
    mask2d = lesion[best_z]

    m, _ = tchebichef_moments(mask2d, TCHEB_N, ROI_SIZE)
    if m is not None:
        all_moments.append(m)

tch_ref = np.mean(np.array(all_moments), axis=0)

# ============================================================
# EXTRACTION FINALE
# ============================================================
rows = []
lesion_id = 1

for lab in labels:
    lesion = (cc_arr == lab).astype(np.uint8)
    best_z = np.argmax([np.sum(lesion[z]) for z in range(lesion.shape[0])])
    mask2d = lesion[best_z]

    tch, bb = tchebichef_moments(mask2d, TCHEB_N, ROI_SIZE)
    if tch is None:
        continue

    recon = reconstruct_from_tchebichef(tch, ROI_SIZE, TCHEB_N)
    recon_bin = binarize_reconstruction(recon)

    imageio.imwrite(
        os.path.join(OUT_RECON_DIR, f"lesion_{lesion_id}_slice_{best_z}.png"),
        (recon * 255).astype(np.uint8)
    )

    dist = shape_distance(tch, tch_ref, SIGMA_DISTANCE)

    rows.append({
        "lesion_id": lesion_id,
        "slice_z": int(best_z),
        "shape_distance": dist,
        **{f"tch_{i}": float(tch[i]) for i in range(len(tch))}
    })

    lesion_id += 1

# ============================================================
# CSV
# ============================================================
fieldnames = ["lesion_id", "slice_z", "shape_distance"] + \
             [f"tch_{i}" for i in range(TCHEB_N * TCHEB_N)]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=CSV_DELIM)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"OK – {len(rows)} lésions traitées")
print(f"Reconstructions : {OUT_RECON_DIR}")
print(f"CSV : {OUT_CSV}")
