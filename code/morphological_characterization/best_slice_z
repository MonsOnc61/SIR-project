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

# Tableau numpy (SimpleITK -> numpy) : (z, y, x)
cc_arr = sitk.GetArrayFromImage(cc).astype(np.int32)

# Remap labels -> lesion_id (1..N)
cc_new = np.zeros_like(cc_arr, dtype=np.int32)
for old_lab, new_id in label_to_newid.items():
    cc_new[cc_arr == old_lab] = new_id

rows = []
n_lesions = len(vols)

for lesion_id in range(1, n_lesions + 1):
    lesion_3d = (cc_new == lesion_id)

    # Best slice Z = max aire (nb pixels)
    areas = lesion_3d.sum(axis=(1, 2))  # sum over (y,x) => per z
    best_z = int(np.argmax(areas))
    best_area = int(areas[best_z])
    if best_area <= 0:
        continue

    # --- centroid sur le best slice (2D) ---
    # coords dans le slice: (y, x)
    ys, xs = np.where(lesion_3d[best_z])
    # centroid voxel (x,y) sur ce slice
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())

    # --- centroid 3D (utile debug) ---
    zs3, ys3, xs3 = np.where(lesion_3d)
    centroid_x_3d = float(xs3.mean())
    centroid_y_3d = float(ys3.mean())
    centroid_z_3d = float(zs3.mean())

    rows.append({
        "lesion_id": int(lesion_id),
        "best_slice_z": int(best_z),

        # centroid 2D sur le best slice (indices voxel)
        "centroid_x": centroid_x,  # x = colonne
        "centroid_y": centroid_y,  # y = ligne
        "centroid_z": float(best_z),

        # centroid 3D (indices voxel)
        "centroid_x_3d": centroid_x_3d,
        "centroid_y_3d": centroid_y_3d,
        "centroid_z_3d": centroid_z_3d,

        "best_area_px": int(best_area),
    })

# Écriture CSV
fieldnames = [
    "lesion_id",
    "best_slice_z",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "centroid_x_3d",
    "centroid_y_3d",
    "centroid_z_3d",
    "best_area_px",
]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=CSV_DELIM)
    writer.writeheader()
    writer.writerows(rows)

print(f"OK: écrit {len(rows)} lésions dans {OUT_CSV}")
print("Rappel: lesion_id = tri par volume 3D décroissant (1 = plus grosse lésion).")
print("Centroids en indices voxel (x,y,z) avec numpy array order (z,y,x).")
