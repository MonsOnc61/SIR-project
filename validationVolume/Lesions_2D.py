import SimpleITK as sitk
import numpy as np
import math
import csv

# =========================
# Réglages
# =========================
MASK_PATH = "MSLesSeg_Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
OUT_CSV = "Output/P1_lesions_2d_features_v2.csv"

FOURIER_K = 12          # nombre de descripteurs de Fourier gardés
LEGENDRE_N = 3          # ordre max Legendre (0..N)
CHEBYSHEV_N = 3         # ordre max Chebyshev (0..N)

CSV_DELIM = ";"         # pratique pour Excel FR

# =========================
# Utils: moments & descriptors
# =========================

def bbox_from_mask(mask2d: np.ndarray):
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)

def perimeter_8conn(mask2d: np.ndarray, spacing_xy=(1.0, 1.0)):
    """
    Périmètre approx en suivant les pixels de contour (8-connexité).
    On calcule la longueur des arêtes entre pixels de contour voisins.
    """
    # contour = pixels mask qui ont au moins un voisin fond
    m = mask2d.astype(np.uint8)
    # voisinage 3x3 via padding
    p = np.pad(m, 1, mode="constant", constant_values=0)
    # somme voisins (incluant lui-même)
    neigh_sum = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    )
    contour = (m == 1) & (neigh_sum < 9)

    # compter transitions contour->fond sur 4-neigh comme approx simple
    # (c'est une approximation, mais stable)
    cx = contour.astype(np.uint8)
    perim_px = (
        np.abs(cx[:, 1:] - cx[:, :-1]).sum() +
        np.abs(cx[1:, :] - cx[:-1, :]).sum()
    )

    sx, sy = spacing_xy
    # approx: transitions horizontales ~ sx, verticales ~ sy
    # on prend une moyenne pondérée simple
    return float(perim_px) * float((sx + sy) / 2.0)

def orientation_eccentricity(mask2d: np.ndarray, spacing_xy=(1.0, 1.0)):
    """
    Orientation (deg) & eccentricity via PCA des coordonnées (en mm).
    Orientation = angle de l'axe principal (major axis) par rapport à l'axe X.
    Eccentricity = sqrt(1 - (lambda2/lambda1)) (ellipse équivalente), entre 0 et 1.
    """
    ys, xs = np.where(mask2d > 0)
    if len(xs) < 3:
        return 0.0, 0.0

    sx, sy = spacing_xy
    X = np.column_stack([xs * sx, ys * sy]).astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)

    C = (X.T @ X) / max(len(X) - 1, 1)
    vals, vecs = np.linalg.eigh(C)  # vals croissantes
    # major = plus grande variance
    v_major = vecs[:, np.argmax(vals)]
    l1 = float(np.max(vals))
    l2 = float(np.min(vals))

    angle = math.degrees(math.atan2(v_major[1], v_major[0]))  # [-180..180]
    # ramener dans [-90..90] pour éviter l'ambiguïté 180°
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180

    ecc = 0.0
    if l1 > 1e-12:
        ecc = math.sqrt(max(0.0, 1.0 - (l2 / l1)))

    return float(angle), float(ecc)

def compactness(area_mm2: float, perimeter_mm: float):
    # circularity / compactness classique
    if perimeter_mm <= 1e-9:
        return 0.0
    return float(4.0 * math.pi * area_mm2 / (perimeter_mm * perimeter_mm))

def fourier_descriptors_from_contour(mask2d: np.ndarray, k=12):
    """
    Descripteurs de Fourier sur le contour.
    On extrait les pixels de contour, on prend leur barycentre et on FFT.
    Retour: k magnitudes normalisées (invariantes à la translation/échelle).
    """
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

    # normalisation par le 1er coefficient non nul (échelle)
    denom = mag[1] if mag.shape[0] > 1 and mag[1] > 1e-12 else (mag.max() if mag.max() > 1e-12 else 1.0)

    # ignorer DC (mag[0]) et prendre k suivants
    feats = []
    for i in range(1, k + 1):
        idx = i % len(mag)
        feats.append(float(mag[idx] / denom))
    return feats

def hu_moments(mask2d: np.ndarray):
    """
    Hu moments (7) sur masque binaire (invariants).
    """
    ys, xs = np.where(mask2d > 0)
    if len(xs) == 0:
        return [0.0]*7

    x = xs.astype(np.float64)
    y = ys.astype(np.float64)

    # moments bruts
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

    # moments normalisés
    def eta(p, q, mu_pq):
        return mu_pq / (m00 ** (1 + (p + q)/2.0))

    n20 = eta(2,0,mu20); n02 = eta(0,2,mu02); n11 = eta(1,1,mu11)
    n30 = eta(3,0,mu30); n03 = eta(0,3,mu03); n21 = eta(2,1,mu21); n12 = eta(1,2,mu12)

    # Hu (formules standard)
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
    # P0=1, P1=x, (n+1)P_{n+1}=(2n+1)xP_n-nP_{n-1}
    if n == 0: return np.ones_like(x)
    if n == 1: return x
    Pnm1 = np.ones_like(x)
    Pn = x
    for k in range(1, n):
        Pnp1 = ((2*k+1)*x*Pn - k*Pnm1) / (k+1)
        Pnm1, Pn = Pn, Pnp1
    return Pn

def chebyshev_poly(n, x):
    # T0=1, T1=x, T_{n+1}=2xT_n - T_{n-1}
    if n == 0: return np.ones_like(x)
    if n == 1: return x
    Tnm1 = np.ones_like(x)
    Tn = x
    for _ in range(1, n):
        Tnp1 = 2*x*Tn - Tnm1
        Tnm1, Tn = Tn, Tnp1
    return Tn

def orthogonal_moments(mask2d: np.ndarray, kind="legendre", N=3):
    """
    Moments orthogonaux faibles ordres sur bbox normalisée en [-1,1]x[-1,1].
    Retour: liste flatten (p,q) pour p,q=0..N.
    """
    bb = bbox_from_mask(mask2d)
    if bb is None:
        return [0.0]*((N+1)*(N+1))
    x0,y0,w,h = bb
    roi = mask2d[y0:y0+h, x0:x0+w].astype(np.float64)

    # grille normalisée [-1,1]
    xs = np.linspace(-1, 1, w, dtype=np.float64)
    ys = np.linspace(-1, 1, h, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)

    feats = []
    for p in range(N+1):
        Px = legendre_poly(p, X) if kind=="legendre" else chebyshev_poly(p, X)
        for q in range(N+1):
            Qy = legendre_poly(q, Y) if kind=="legendre" else chebyshev_poly(q, Y)
            m = np.sum(roi * Px * Qy)
            feats.append(float(m))
    # normalisation simple
    denom = max(1.0, float(np.sum(roi)))
    feats = [v/denom for v in feats]
    return feats

# =========================
# 1) Charger + binariser
# =========================
mask_img = sitk.ReadImage(MASK_PATH)
mask_bin = sitk.BinaryThreshold(mask_img, 0.5, 1e9, 1, 0)
mask_bin = sitk.Cast(mask_bin, sitk.sitkUInt8)

spacing = mask_bin.GetSpacing()  # (sx,sy,sz)
sx, sy, sz = spacing

# =========================
# 2) Lésions 3D + tri volume décroissant pour ID stable
# =========================
cc = sitk.ConnectedComponent(mask_bin)  # labels 1..N

shape3d = sitk.LabelShapeStatisticsImageFilter()
shape3d.Execute(cc)
orig_labels = list(shape3d.GetLabels())

if len(orig_labels) == 0:
    raise RuntimeError("Aucune lésion dans le masque.")

# tri par volume décroissant
vols = [(lab, shape3d.GetPhysicalSize(lab)) for lab in orig_labels]
vols.sort(key=lambda x: x[1], reverse=True)

# mapping ancien_label -> new_id (1..)
label_to_newid = {lab: i+1 for i,(lab,_) in enumerate(vols)}

# On convertit cc en array pour remapper proprement
cc_arr = sitk.GetArrayFromImage(cc)  # (z,y,x)
cc_new = np.zeros_like(cc_arr, dtype=np.int32)
for old_lab, new_id in label_to_newid.items():
    cc_new[cc_arr == old_lab] = new_id

cc_new_img = sitk.GetImageFromArray(cc_new)
cc_new_img.CopyInformation(cc)

# stats à nouveau sur nouveaux IDs
shape3d_new = sitk.LabelShapeStatisticsImageFilter()
shape3d_new.Execute(sitk.Cast(cc_new_img, sitk.sitkUInt16))

# =========================
# 3) Pour chaque lésion: choisir best slice = max area 2D
# =========================
mask_arr = sitk.GetArrayFromImage(mask_bin)  # (z,y,x)

rows = []
for lesion_id in range(1, len(vols)+1):
    # voxels de la lésion
    lesion_3d = (cc_new == lesion_id).astype(np.uint8)

    # best slice = max nombre de pixels
    best_z = None
    best_area_px = -1
    for z in range(lesion_3d.shape[0]):
        a = int(np.sum(lesion_3d[z]))
        if a > best_area_px:
            best_area_px = a
            best_z = z

    if best_z is None or best_area_px <= 0:
        continue

    mask2d = lesion_3d[best_z]  # (y,x)

    # =========================
    # 4) Features 2D
    # =========================
    area_mm2 = float(best_area_px) * sx * sy
    perim_mm = perimeter_8conn(mask2d, (sx,sy))
    comp = compactness(area_mm2, perim_mm)
    orient_deg, ecc = orientation_eccentricity(mask2d, (sx,sy))

    bb = bbox_from_mask(mask2d)
    if bb is None:
        x0=y0=w=h=0
    else:
        x0,y0,w,h = bb

    # centroid 2D en mm
    ys, xs = np.where(mask2d > 0)
    cx_mm = float(xs.mean()*sx)
    cy_mm = float(ys.mean()*sy)

    # Fourier (K)
    fourier = fourier_descriptors_from_contour(mask2d, FOURIER_K)

    # Hu moments (7)
    hu = hu_moments(mask2d)

    # Legendre & Chebyshev moments
    leg = orthogonal_moments(mask2d, kind="legendre", N=LEGENDRE_N)
    cheb = orthogonal_moments(mask2d, kind="chebyshev", N=CHEBYSHEV_N)

    rows.append({
        "lesion_id": lesion_id,
        "best_slice_z": int(best_z),
        "area_mm2": area_mm2,
        "perimeter_mm": perim_mm,
        "compactness": comp,
        "eccentricity": ecc,
        "orientation_deg": orient_deg,
        "centroid_x_mm": cx_mm,
        "centroid_y_mm": cy_mm,
        "bbox_x": int(x0),
        "bbox_y": int(y0),
        "bbox_w": int(w),
        "bbox_h": int(h),
        **{f"fourier_{i+1}": fourier[i] for i in range(FOURIER_K)},
        **{f"hu_{i+1}": hu[i] for i in range(7)},
        **{f"leg_{i}": leg[i] for i in range(len(leg))},
        **{f"cheb_{i}": cheb[i] for i in range(len(cheb))},
    })

# =========================
# 5) Écriture CSV (Excel-friendly)
# =========================
# colonnes dans un ordre stable
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
