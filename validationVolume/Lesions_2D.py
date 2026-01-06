import os
import math
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# =========================
# Réglages
# =========================
MASK_PATH = "MSLesSeg_Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
OUT_CSV = "Output/P1_lesions_2d_features_v3.csv"

FOURIER_K = 12          # nombre de descripteurs de Fourier gardés
LEGENDRE_N = 3          # ordre max Legendre (0..N)
CHEBYSHEV_N = 3         # ordre max Chebyshev (0..N)

# =========================
# (Optionnel) ICIP-like VRG + Shape prior (Tchebichef)
# =========================
USE_VRG_REFINEMENT = False     # mets True pour activer le raffinement ICIP
IMG_PATH = None               # si None: on tente MASK_PATH.replace("_MASK","")

VRG_ETA = 20
VRG_SIGMA = 1.5
VRG_ALPHA = 200.0
VRG_MAX_ITER = 120
VRG_USE_SIMPLE_NORM = False
VRG_ROI_MARGIN = 10

CSV_DELIM = ";"  # Excel FR


# =========================
# ICIP-like VRG with Tchebichef shape prior (2D)
# =========================

def _comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    k = min(k, n - k)
    res = 1.0
    for i in range(1, k + 1):
        res *= (n - (k - i)) / i
    return res

def tchebichef_poly(p: int, N: int) -> np.ndarray:
    x = np.arange(N, dtype=np.float64)
    tp = np.zeros(N, dtype=np.float64)

    pfact = 1.0
    for i in range(2, p + 1):
        pfact *= i
    Np = float(N) ** p if p > 0 else 1.0

    for k in range(0, p + 1):
        sign = -1.0 if ((p - k) % 2 == 1) else 1.0
        a = _comb(N - 1 - k, p - k)
        b = _comb(p + k, p)

        if k == 0:
            cxk = np.ones_like(x)
        else:
            numer = np.ones_like(x)
            for j in range(k):
                numer *= (x - j)
            kfact = 1.0
            for i in range(2, k + 1):
                kfact *= i
            cxk = numer / kfact

        tp += sign * a * b * cxk

    tp *= pfact / Np
    return tp

def rho_tcheb(p: int, N: int) -> float:
    if p == 0:
        return float(N)
    r_prev = rho_tcheb(p - 1, N)
    num = (N**2 - p**2) * (N**2)
    den = (2 * p - 1) * (2 * p + 1)
    return (num / den) * r_prev

def Cpq_tcheb(p: int, q: int, N: int) -> float:
    return 1.0 / (rho_tcheb(p, N) * rho_tcheb(q, N))

def Hpq_sigma(p: int, q: int, sigma: float) -> float:
    return (1.0 / np.sqrt(2.0 * np.pi * sigma**2)) * np.exp(-((p + q) ** 2) / (2.0 * sigma**2))

def compute_tcheb_basis(N: int, eta: int) -> np.ndarray:
    tp = np.zeros((eta, N), dtype=np.float64)
    for p in range(eta):
        tp[p] = tchebichef_poly(p, N)
    return tp

def moments_tchebichef(mask_inside01: np.ndarray, tp: np.ndarray, eta: int, use_simple_norm: bool = False):
    H, W = mask_inside01.shape
    N = max(H, W)

    phi = np.zeros((N, N), dtype=np.float64)
    phi[:H, :W] = mask_inside01.astype(np.float64)

    if use_simple_norm:
        ys, xs = np.where(phi > 0.5)
        if len(xs) > 10:
            cx, cy = xs.mean(), ys.mean()
            sx, sy = xs.std() + 1e-6, ys.std() + 1e-6
            scale = N / 6.0
            uxs = (xs - cx) / sx * scale + (N - 1) / 2.0
            uys = (ys - cy) / sy * scale + (N - 1) / 2.0
            uxs = np.clip(np.round(uxs).astype(int), 0, N - 1)
            uys = np.clip(np.round(uys).astype(int), 0, N - 1)
            phi2 = np.zeros_like(phi)
            phi2[uys, uxs] = 1.0
            phi = phi2

    T = np.zeros((eta, eta), dtype=np.float64)
    for p in range(eta):
        for q in range(eta):
            cpq = Cpq_tcheb(p, q, N)
            T[p, q] = cpq * (tp[q] @ phi @ tp[p])
    return T, N

def build_candidates_4n(phi01: np.ndarray) -> np.ndarray:
    H, W = phi01.shape
    inside = (phi01 == 0)
    cand = np.zeros_like(phi01, dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        y0 = max(0, -dy); y1 = min(H, H - dy)
        x0 = max(0, -dx); x1 = min(W, W - dx)
        a = inside[y0:y1, x0:x1]
        b = inside[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        edge = (a != b)
        cand[y0:y1, x0:x1] |= edge
        cand[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= edge
    return cand

def compute_means_phi(I: np.ndarray, phi01: np.ndarray):
    inside = (phi01 == 0)
    outside = ~inside
    mu_in = float(I[inside].mean()) if inside.any() else float(I.mean())
    mu_out = float(I[outside].mean()) if outside.any() else float(I.mean())
    return mu_in, mu_out

def vrg_shape_prior_2d(
    I: np.ndarray,
    seed_inside: np.ndarray,
    ref_inside: np.ndarray,
    eta: int = 20,
    sigma: float = 1.5,
    alpha: float = 200.0,
    max_iter: int = 120,
    use_simple_norm: bool = False,
    verbose: bool = False,
):
    I = I.astype(np.float64)
    H, W = I.shape

    phi = np.ones((H, W), dtype=np.int8)     # outside=1
    phi[seed_inside.astype(bool)] = 0        # inside=0

    N = max(H, W)
    tp = compute_tcheb_basis(N, eta)

    Tref, _ = moments_tchebichef(ref_inside.astype(np.uint8), tp, eta, use_simple_norm=use_simple_norm)
    Wpq = np.zeros((eta, eta), dtype=np.float64)
    for p in range(eta):
        for q in range(eta):
            Wpq[p, q] = Hpq_sigma(p, q, sigma) ** 2

    cur_inside = (phi == 0).astype(np.uint8)
    Tn, _ = moments_tchebichef(cur_inside, tp, eta, use_simple_norm=use_simple_norm)

    mu_in, mu_out = compute_means_phi(I, phi)

    def Rv_at(vy: int, vx: int) -> np.ndarray:
        R = np.zeros((eta, eta), dtype=np.float64)
        for p in range(eta):
            for q in range(eta):
                R[p, q] = Cpq_tcheb(p, q, N) * tp[p, vx] * tp[q, vy]
        return R

    for it in range(max_iter):
        cand = build_candidates_4n(phi)
        ys, xs = np.where(cand)
        if len(xs) == 0:
            break

        changed = 0
        for vy, vx in zip(ys, xs):
            phi_v = int(phi[vy, vx])
            s = (1.0 - 2.0 * phi_v)

            dJI = s * (((I[vy, vx] - mu_in) ** 2) - ((I[vy, vx] - mu_out) ** 2))

            Rv = Rv_at(vy, vx)
            term = (2.0 * Tn * Rv) + (s * (Rv * Rv)) - (2.0 * Tref * Rv)
            dJprior = s * float(np.sum(Wpq * term))

            dJT = dJI + alpha * dJprior

            if dJT < 0.0:
                phi[vy, vx] = 1 - phi[vy, vx]
                changed += 1
                Tn = Tn + s * Rv

        mu_in, mu_out = compute_means_phi(I, phi)

        if verbose:
            print(f"[VRG] iter={it} changed={changed} area={(phi==0).sum()}")

        if changed == 0:
            break

    return (phi == 0), phi.astype(np.uint8)

def _resize_mask_nn(mask: np.ndarray, out_shape):
    in_h, in_w = mask.shape
    out_h, out_w = out_shape
    if in_h == out_h and in_w == out_w:
        return mask.astype(np.uint8)
    yy = (np.linspace(0, in_h - 1, out_h)).round().astype(int)
    xx = (np.linspace(0, in_w - 1, out_w)).round().astype(int)
    return mask[np.ix_(yy, xx)].astype(np.uint8)

def bbox_from_mask(mask2d: np.ndarray):
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)

def _crop_with_margin(mask2d: np.ndarray, margin: int):
    bb = bbox_from_mask(mask2d)
    if bb is None:
        return None
    x, y, w, h = bb
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(mask2d.shape[1], x + w + margin)
    y1 = min(mask2d.shape[0], y + h + margin)
    return x0, y0, x1, y1

def perimeter_8conn(mask2d: np.ndarray, spacing_xy=(1.0, 1.0)):
    m = mask2d.astype(np.uint8)
    p = np.pad(m, 1, mode="constant", constant_values=0)
    neigh_sum = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    contour = (m == 1) & (neigh_sum < 9)
    cx = contour.astype(np.uint8)
    perim_px = (
        np.abs(cx[:, 1:] - cx[:, :-1]).sum() +
        np.abs(cx[1:, :] - cx[:-1, :]).sum()
    )
    sx, sy = spacing_xy
    return float(perim_px) * float((sx + sy) / 2.0)

def orientation_eccentricity(mask2d: np.ndarray, spacing_xy=(1.0, 1.0)):
    ys, xs = np.where(mask2d > 0)
    if len(xs) < 3:
        return 0.0, 0.0
    sx, sy = spacing_xy
    X = np.column_stack([xs * sx, ys * sy]).astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)
    C = (X.T @ X) / max(len(X) - 1, 1)
    vals, vecs = np.linalg.eigh(C)
    v_major = vecs[:, np.argmax(vals)]
    l1 = float(np.max(vals))
    l2 = float(np.min(vals))
    angle = math.degrees(math.atan2(v_major[1], v_major[0]))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    ecc = 0.0
    if l1 > 1e-12:
        ecc = math.sqrt(max(0.0, 1.0 - (l2 / l1)))
    return float(angle), float(ecc)

def compactness(area_mm2: float, perimeter_mm: float):
    if perimeter_mm <= 1e-9:
        return 0.0
    return float(4.0 * math.pi * area_mm2 / (perimeter_mm * perimeter_mm))

def fourier_descriptors_from_contour(mask2d: np.ndarray, k=12):
    m = mask2d.astype(np.uint8)
    p = np.pad(m, 1, mode="constant", constant_values=0)
    neigh_sum = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    contour = (m == 1) & (neigh_sum < 9)
    ys, xs = np.where(contour)
    if len(xs) < 8:
        return [0.0] * k

    z = xs.astype(np.float64) + 1j * ys.astype(np.float64)
    z = z - np.mean(z)
    F = np.fft.fft(z)
    mag = np.abs(F)
    denom = mag[1] if mag.shape[0] > 1 and mag[1] > 1e-12 else (mag.max() if mag.max() > 1e-12 else 1.0)

    feats = []
    for i in range(1, k + 1):
        idx = i % len(mag)
        feats.append(float(mag[idx] / denom))
    return feats

def hu_moments(mask2d: np.ndarray):
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return [0.0] * 7

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    m00 = len(xs)
    xbar = x.mean()
    ybar = y.mean()
    x_ = x - xbar
    y_ = y - ybar

    mu20 = np.sum(x_**2)
    mu02 = np.sum(y_**2)
    mu11 = np.sum(x_*y_)
    mu30 = np.sum(x_**3)
    mu03 = np.sum(y_**3)
    mu21 = np.sum(x_**2 * y_)
    mu12 = np.sum(x_ * y_**2)

    def eta(p, q, mu_pq):
        return mu_pq / (m00 ** (1 + (p + q) / 2.0))

    n20 = eta(2,0,mu20); n02 = eta(0,2,mu02); n11 = eta(1,1,mu11)
    n30 = eta(3,0,mu30); n03 = eta(0,3,mu03); n21 = eta(2,1,mu21); n12 = eta(1,2,mu12)

    h1 = n20 + n02
    h2 = (n20 - n02)**2 + 4*n11**2
    h3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    h4 = (n30 + n12)**2 + (n21 + n03)**2
    h5 = (n30 - 3*n12)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) + \
         (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)
    h6 = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) + 4*n11*(n30 + n12)*(n21 + n03)
    h7 = (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) - \
         (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)

    return [float(h) for h in (h1,h2,h3,h4,h5,h6,h7)]

def legendre_poly(n, x):
    if n == 0: return np.ones_like(x)
    if n == 1: return x
    Pnm1 = np.ones_like(x)
    Pn = x
    for k in range(1, n):
        Pnp1 = ((2*k+1)*x*Pn - k*Pnm1) / (k+1)
        Pnm1, Pn = Pn, Pnp1
    return Pn

def chebyshev_poly(n, x):
    if n == 0: return np.ones_like(x)
    if n == 1: return x
    Tnm1 = np.ones_like(x)
    Tn = x
    for _ in range(1, n):
        Tnp1 = 2*x*Tn - Tnm1
        Tnm1, Tn = Tn, Tnp1
    return Tn

def orthogonal_moments(mask2d: np.ndarray, kind="legendre", N=3):
    bb = bbox_from_mask(mask2d)
    if bb is None:
        return [0.0] * ((N+1)*(N+1))
    x0,y0,w,h = bb
    roi = mask2d[y0:y0+h, x0:x0+w].astype(np.float64)

    xs = np.linspace(-1, 1, w, dtype=np.float64)
    ys = np.linspace(-1, 1, h, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)

    feats = []
    for p in range(N+1):
        Px = legendre_poly(p, X) if kind=="legendre" else chebyshev_poly(p, X)
        for q in range(N+1):
            Qy = legendre_poly(q, Y) if kind=="legendre" else chebyshev_poly(q, Y)
            feats.append(float(np.sum(roi * Px * Qy)))

    denom = max(1.0, float(np.sum(roi)))
    feats = [v/denom for v in feats]
    return feats


# =========================
# MAIN
# =========================

os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)

mask_img = sitk.ReadImage(MASK_PATH)
mask_bin = sitk.BinaryThreshold(mask_img, 0.5, 1e9, 1, 0)
mask_bin = sitk.Cast(mask_bin, sitk.sitkUInt8)

spacing = mask_bin.GetSpacing()  # (sx,sy,sz)
sx, sy, sz = spacing

# Optionnel: image intensité pour VRG
img_arr = None
if USE_VRG_REFINEMENT:
    _img_path = IMG_PATH
    if _img_path is None and "_MASK" in MASK_PATH:
        _img_path = MASK_PATH.replace("_MASK", "")
    if _img_path is None or (not os.path.exists(_img_path)):
        print(f"WARNING: IMG_PATH introuvable ({_img_path}). VRG désactivé.")
        USE_VRG_REFINEMENT = False
    else:
        img_itk = sitk.ReadImage(_img_path)
        img_arr = sitk.GetArrayFromImage(img_itk).astype(np.float64)  # (z,y,x)

# 3D connected components
cc = sitk.ConnectedComponent(mask_bin)  # labels 1..N
shape3d = sitk.LabelShapeStatisticsImageFilter()
shape3d.Execute(cc)
orig_labels = list(shape3d.GetLabels())
if len(orig_labels) == 0:
    raise RuntimeError("Aucune lésion dans le masque.")

# tri par volume décroissant => IDs stables
vols = [(lab, shape3d.GetPhysicalSize(lab)) for lab in orig_labels]
vols.sort(key=lambda x: x[1], reverse=True)
label_to_newid = {lab: i+1 for i, (lab, _) in enumerate(vols)}

cc_arr = sitk.GetArrayFromImage(cc)  # (z,y,x)
cc_new = np.zeros_like(cc_arr, dtype=np.int32)
for old_lab, new_id in label_to_newid.items():
    cc_new[cc_arr == old_lab] = new_id

# Référence VRG: plus grosse lésion (slice 2D max)
_REF_MASK2D = None
if USE_VRG_REFINEMENT and img_arr is not None:
    lesion1 = (cc_new == 1).astype(np.uint8)
    if lesion1.any():
        best_z_ref = int(np.argmax([lesion1[z].sum() for z in range(lesion1.shape[0])]))
        _REF_MASK2D = lesion1[best_z_ref].astype(np.uint8)

rows = []

for lesion_id in range(1, len(vols) + 1):
    lesion_3d = (cc_new == lesion_id).astype(np.uint8)

    # best slice = max area
    areas = [int(lesion_3d[z].sum()) for z in range(lesion_3d.shape[0])]
    best_z = int(np.argmax(areas))
    best_area_px = int(areas[best_z])
    if best_area_px <= 0:
        continue

    mask2d = lesion_3d[best_z].astype(np.uint8)  # (y,x)

    # ============ VRG refinement (optionnel) ============
    if USE_VRG_REFINEMENT and (img_arr is not None) and (_REF_MASK2D is not None):
        crop = _crop_with_margin(mask2d, VRG_ROI_MARGIN)
        if crop is not None:
            x0r, y0r, x1r, y1r = crop
            I2d = img_arr[best_z, y0r:y1r, x0r:x1r]
            seed_roi = mask2d[y0r:y1r, x0r:x1r].astype(np.uint8)
            ref_roi = _resize_mask_nn(_REF_MASK2D, seed_roi.shape)

            try:
                seg_roi, _ = vrg_shape_prior_2d(
                    I2d,
                    seed_inside=seed_roi,
                    ref_inside=ref_roi,
                    eta=VRG_ETA,
                    sigma=VRG_SIGMA,
                    alpha=VRG_ALPHA,
                    max_iter=VRG_MAX_ITER,
                    use_simple_norm=VRG_USE_SIMPLE_NORM,
                    verbose=False,
                )
                mask2d2 = mask2d.copy()
                mask2d2[y0r:y1r, x0r:x1r] = seg_roi.astype(np.uint8)
                mask2d = mask2d2
            except Exception as e:
                print(f"WARNING: VRG échoué lesion_id={lesion_id} slice={best_z}: {e}")

    # ============ Features (toujours calculées) ============
    area_mm2 = float(best_area_px) * sx * sy
    perim_mm = perimeter_8conn(mask2d, (sx, sy))
    comp = compactness(area_mm2, perim_mm)
    orient_deg, ecc = orientation_eccentricity(mask2d, (sx, sy))

    bb = bbox_from_mask(mask2d)
    if bb is None:
        x0 = y0 = w = h = 0
        cx_mm = cy_mm = 0.0
    else:
        x0, y0, w, h = bb
        ys, xs = np.where(mask2d > 0)
        cx_mm = float(xs.mean() * sx)
        cy_mm = float(ys.mean() * sy)

    fourier = fourier_descriptors_from_contour(mask2d, FOURIER_K)
    hu = hu_moments(mask2d)
    leg = orthogonal_moments(mask2d, kind="legendre", N=LEGENDRE_N)
    cheb = orthogonal_moments(mask2d, kind="chebyshev", N=CHEBYSHEV_N)

    rows.append({
        "lesion_id": int(lesion_id),
        "best_slice_z": int(best_z),
        "area_mm2": float(area_mm2),
        "perimeter_mm": float(perim_mm),
        "compactness": float(comp),
        "eccentricity": float(ecc),
        "orientation_deg": float(orient_deg),
        "centroid_x_mm": float(cx_mm),
        "centroid_y_mm": float(cy_mm),
        "bbox_x": int(x0),
        "bbox_y": int(y0),
        "bbox_w": int(w),
        "bbox_h": int(h),
        **{f"fourier_{i+1}": float(fourier[i]) for i in range(FOURIER_K)},
        **{f"hu_{i+1}": float(hu[i]) for i in range(7)},
        **{f"leg_{i}": float(leg[i]) for i in range(len(leg))},
        **{f"cheb_{i}": float(cheb[i]) for i in range(len(cheb))},
    })

# ============ Écriture CSV ============
base_cols = [
    "lesion_id","best_slice_z","area_mm2","perimeter_mm",
    "compactness","eccentricity","orientation_deg",
    "centroid_x_mm","centroid_y_mm",
    "bbox_x","bbox_y","bbox_w","bbox_h",
]
fourier_cols = [f"fourier_{i+1}" for i in range(FOURIER_K)]
hu_cols = [f"hu_{i+1}" for i in range(7)]
leg_cols = [f"leg_{i}" for i in range((LEGENDRE_N+1)*(LEGENDRE_N+1))]
cheb_cols = [f"cheb_{i}" for i in range((CHEBYSHEV_N+1)*(CHEBYSHEV_N+1))]
fieldnames = base_cols + fourier_cols + hu_cols + leg_cols + cheb_cols

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=CSV_DELIM)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"OK: écrit {len(rows)} lésions dans {OUT_CSV}")
print("Rappel: lesion_id = tri par volume 3D décroissant (stable pour tout le groupe).")
