import os
import csv
import numpy as np
import SimpleITK as sitk

# =========================
# Réglages
# =========================
MASK_PATH = "MSLesSeg_Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
OUT_CSV   = "Output/P1_best_slice_per_lesion.csv"
CSV_DELIM = ";"

# =========================
# MAIN
# =========================
os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)

mask_img = sitk.ReadImage(MASK_PATH)

# Binarisation: tout ce qui > 0 devient 1
mask_bin = sitk.BinaryThreshold(
    mask_img,
    lowerThreshold=0.5,
    upperThreshold=1e9,
    insideValue=1,
    outsideValue=0
)
mask_bin = sitk.Cast(mask_bin, sitk.sitkUInt8)

# Connected components 3D (labels 1..N)
cc = sitk.ConnectedComponent(mask_bin)

shape3d = sitk.LabelShapeStatisticsImageFilter()
shape3d.Execute(cc)
labels = list(shape3d.GetLabels())
if len(labels) == 0:
    raise RuntimeError("Aucune lésion dans le masque.")

# Tri par volume décroissant => lesion_id stable (1 = plus grosse)
vols = [(lab, shape3d.GetPhysicalSize(lab)) for lab in labels]
vols.sort(key=lambda x: x[1], reverse=True)
label_to_newid = {lab: i + 1 for i, (lab, _) in enumerate(vols)}
newid_to_label = {new_id: old_lab for old_lab, new_id in label_to_newid.items()}

# Tableau numpy (SimpleITK -> numpy) : (z, y, x)
cc_arr = sitk.GetArrayFromImage(cc).astype(np.int32)

# Remap labels -> lesion_id (1..N)
cc_new = np.zeros_like(cc_arr, dtype=np.int32)
for old_lab, new_id in label_to_newid.items():
    cc_new[cc_arr == old_lab] = new_id

def compute_flatness_from_shape_stats(shape_filter, lab):
    """
    Flatness ITK/SimpleITK (si dispo) : ratio basé sur les moments principaux.
    On essaie:
      1) GetFlatness si disponible
      2) sinon via GetPrincipalMoments (m1>=m2>=m3) : sqrt(m3/m1)
      3) sinon NaN
    """
    # 1) Direct
    if hasattr(shape_filter, "GetFlatness"):
        try:
            return float(shape_filter.GetFlatness(lab))
        except Exception:
            pass

    # 2) Par moments
    if hasattr(shape_filter, "GetPrincipalMoments"):
        try:
            m = shape_filter.GetPrincipalMoments(lab)  # tuple/list
            m = sorted([float(x) for x in m], reverse=True)  # m1>=m2>=m3
            if len(m) >= 3 and m[0] > 0:
                return float(np.sqrt(m[2] / m[0]))
        except Exception:
            pass

    return float("nan")

rows = []
n_lesions = len(vols)

for lesion_id in range(1, n_lesions + 1):
    old_lab = newid_to_label[lesion_id]
    lesion_3d = (cc_new == lesion_id)

    # =========================
    # Best slices (Z, Y, X)
    # =========================

    # --- Z (plan XY) ---
    areas_z = lesion_3d.sum(axis=(1, 2))  # (z)
    best_z = int(np.argmax(areas_z))
    best_area_z = int(areas_z[best_z])

    # --- Y (plan XZ) ---
    areas_y = lesion_3d.sum(axis=(0, 2))  # (y)
    best_y = int(np.argmax(areas_y))
    best_area_y = int(areas_y[best_y])

    # --- X (plan YZ) ---
    areas_x = lesion_3d.sum(axis=(0, 1))  # (x)
    best_x = int(np.argmax(areas_x))
    best_area_x = int(areas_x[best_x])

    if best_area_z <= 0:
        continue

    # =========================
    # Centroids
    # =========================

    # --- centroid 2D sur best Z ---
    ys, xs = np.where(lesion_3d[best_z])
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    # --- centroid 3D ---
    zs3, ys3, xs3 = np.where(lesion_3d)
    centroid_x_3d = float(xs3.mean())
    centroid_y_3d = float(ys3.mean())
    centroid_z_3d = float(zs3.mean())

    # =========================
    # Flatness + Density
    # =========================
    flatness_3d = compute_flatness_from_shape_stats(shape3d, old_lab)

    bbox = shape3d.GetBoundingBox(old_lab)  # (x,y,z,sizeX,sizeY,sizeZ)
    sizeX, sizeY, sizeZ = int(bbox[3]), int(bbox[4]), int(bbox[5])
    bbox_vox = max(1, sizeX * sizeY * sizeZ)
    nvox = int(shape3d.GetNumberOfPixels(old_lab))
    density_3d = float(nvox / bbox_vox)

    rows.append({
        "lesion_id": lesion_id,

        # best slices
        "best_slice_z": best_z,
        "best_slice_y": best_y,
        "best_slice_x": best_x,

        # centroid 2D (sur best Z)
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "centroid_z": float(best_z),

        # centroid 3D
        "centroid_x_3d": centroid_x_3d,
        "centroid_y_3d": centroid_y_3d,
        "centroid_z_3d": centroid_z_3d,

        # areas
        "best_area_z_px": best_area_z,
        "best_area_y_px": best_area_y,
        "best_area_x_px": best_area_x,

        # shape
        "flatness_3d": flatness_3d,
        "density_3d": density_3d,
    })

# Écriture CSV
fieldnames = [
    "lesion_id",

    "best_slice_z",
    "best_slice_y",
    "best_slice_x",

    "centroid_x",
    "centroid_y",
    "centroid_z",

    "centroid_x_3d",
    "centroid_y_3d",
    "centroid_z_3d",

    "best_area_z_px",
    "best_area_y_px",
    "best_area_x_px",

    "flatness_3d",
    "density_3d",
]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=CSV_DELIM)
    writer.writeheader()
    writer.writerows(rows)

print(f"OK: écrit {len(rows)} lésions dans {OUT_CSV}")
print("Rappel: lesion_id = tri par volume 3D décroissant (1 = plus grosse lésion).")
print("Centroids en indices voxel (x,y,z) avec numpy array order (z,y,x).")
print("flatness_3d: (si dispo) ITK/SimpleITK Flatness, sinon sqrt(m3/m1) via moments principaux.")
print("density_3d: nb_voxels / (sizeX*sizeY*sizeZ) de la bounding box 3D (en voxels).")
