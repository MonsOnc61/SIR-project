import SimpleITK as sitk
import math

# ============
# Entrées/sorties
# ============
mask_path = "MSLesSeg_Dataset/train/P1/T1/P1_T1_MASK.nii.gz"  # masque binaire 0/1
out_label3d_path = "Output/lesions_labeled_3d_sorted.nii.gz"

# ============
# 1) Lecture + binarisation sûre
# ============
mask = sitk.ReadImage(mask_path)
mask_bin = sitk.BinaryThreshold(mask, 0.5, 1e9, 1, 0)
mask_bin = sitk.Cast(mask_bin, sitk.sitkUInt8)

# ============
# 2) Labellisation 3D + tri volume décroissant
#    lesion 1 = plus grosse lésion en volume
# ============
cc3d = sitk.ConnectedComponent(mask_bin)  # labels 1..N (non triés)
label3d = sitk.RelabelComponent(cc3d, sortByObjectSize=True)  # tri par taille décroissante

sitk.WriteImage(label3d, out_label3d_path)
print(f"[OK] Label 3D trié écrit : {out_label3d_path}")

# ============
# 3) Choisir la coupe axiale (z) avec surface totale max
#    (on reste en 2D pour les features)
# ============
size = label3d.GetSize()  # (x, y, z)
best_k = 0
best_sum = -1.0

# On calcule sur un masque 2D binaire "il y a lésion ou non"
for k in range(size[2]):
    extract = sitk.ExtractImageFilter()
    extract.SetSize([size[0], size[1], 0])      # sortir une image 2D
    extract.SetIndex([0, 0, k])
    slice_labels = extract.Execute(label3d)

    slice_bin = sitk.NotEqual(slice_labels, 0)
    slice_bin = sitk.Cast(slice_bin, sitk.sitkUInt8)

    stats = sitk.StatisticsImageFilter()
    stats.Execute(slice_bin)
    s = stats.GetSum()  # nb de pixels lésions sur la coupe (surface en pixels)

    if s > best_sum:
        best_sum = s
        best_k = k

print(f"[INFO] Coupe axiale choisie (surface lésions max) : k={best_k} (pixels lésions={int(best_sum)})")

# ============
# 4) Extraction de la coupe 2D labellisée (IDs cohérentes)
# ============
extract = sitk.ExtractImageFilter()
extract.SetSize([size[0], size[1], 0])
extract.SetIndex([0, 0, best_k])
slice_labels = extract.Execute(label3d)  # 2D, labels = IDs 3D

# ============
# 5) Mesures 2D par lésion sur la coupe
# ============
shape2d = sitk.LabelShapeStatisticsImageFilter()
shape2d.Execute(slice_labels)

labels_2d = list(shape2d.GetLabels())
labels_2d.sort()

print("\n=== Features 2D sur la coupe ===")
print("(IDs = celles du 3D trié : 1 = plus grosse lésion en volume total)\n")

for lab in labels_2d:
    if lab == 0:
        continue

    area_mm2 = shape2d.GetPhysicalSize(lab)   # en 2D = aire en mm^2
    centroid = shape2d.GetCentroid(lab)       # (x,y) en mm
    bbox = shape2d.GetBoundingBox(lab)        # (x,y, size_x, size_y)
    elong = shape2d.GetElongation(lab)
    flat = shape2d.GetFlatness(lab)

    # Périmètre (dispo en 2D)
    try:
        perim = shape2d.GetPerimeter(lab)
    except Exception:
        perim = None

    # Hu moments (invariants de forme) si dispo
    try:
        hu = shape2d.GetHuMoments(lab)  # tuple de 7 valeurs
    except Exception:
        hu = None

    print(f"Lésion {lab}:")
    print(f"  Aire (mm^2)        : {area_mm2:.2f}")
    if perim is not None:
        print(f"  Périmètre (mm)     : {perim:.2f}")
    print(f"  Centroid (mm)      : {centroid}")
    print(f"  BBox (index+taille): {bbox}")
    print(f"  Elongation         : {elong:.3f}")
    print(f"  Flatness           : {flat:.3f}")
    if hu is not None:
        print(f"  Hu moments         : {[float(x) for x in hu]}")
    print()
