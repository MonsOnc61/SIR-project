import SimpleITK as sitk
import math

# ======================
# 1) Chemins à adapter
# ======================
mask_T1_path = "MSLesSeg_Dataset/train/P1/T1/P1_T1_MASK.nii.gz"
mask_T2_path = "MSLesSeg_Dataset/train/P1/T2/P1_T2_MASK.nii.gz"

# ==========================
# 2) Lecture des deux masques
# ==========================
mask_T1_img = sitk.ReadImage(mask_T1_path)
mask_T2_img = sitk.ReadImage(mask_T2_path)

print("T1 mask size :", mask_T1_img.GetSize())
print("T2 mask size :", mask_T2_img.GetSize())
print()

# Vérifier que T1 et T2 sont alignés (même espace)
same_size      = (mask_T1_img.GetSize()      == mask_T2_img.GetSize())
same_origin    = (mask_T1_img.GetOrigin()    == mask_T2_img.GetOrigin())
same_spacing   = (mask_T1_img.GetSpacing()   == mask_T2_img.GetSpacing())
same_direction = (mask_T1_img.GetDirection() == mask_T2_img.GetDirection())

if not (same_size and same_origin and same_spacing and same_direction):
    raise RuntimeError("Les masques T1 et T2 ne sont PAS alignés (taille/origin/spacing/direction différents).")


# ==========================================
# 3) Binarisation + cast en UInt8 (0/1 entier)
#    => corrige l'erreur ConnectedComponent
# ==========================================
mask_T1_bin = sitk.BinaryThreshold(mask_T1_img,
                                   lowerThreshold=0.5,
                                   upperThreshold=1e9,
                                   insideValue=1,
                                   outsideValue=0)
mask_T1_bin = sitk.Cast(mask_T1_bin, sitk.sitkUInt8)

mask_T2_bin = sitk.BinaryThreshold(mask_T2_img,
                                   lowerThreshold=0.5,
                                   upperThreshold=1e9,
                                   insideValue=1,
                                   outsideValue=0)
mask_T2_bin = sitk.Cast(mask_T2_bin, sitk.sitkUInt8)


# =======================================
# 4) Composantes connexes à T1 et à T2
# =======================================
cc_T1 = sitk.ConnectedComponent(mask_T1_bin)
cc_T2 = sitk.ConnectedComponent(mask_T2_bin)

stats_T1 = sitk.LabelShapeStatisticsImageFilter()
stats_T1.Execute(cc_T1)

stats_T2 = sitk.LabelShapeStatisticsImageFilter()
stats_T2.Execute(cc_T2)

labels_T1 = list(stats_T1.GetLabels())
labels_T2 = list(stats_T2.GetLabels())
labels_T1.sort()
labels_T2.sort()

print(f"Nombre de lésions à T1 : {len(labels_T1)}")
print(f"Nombre de lésions à T2 : {len(labels_T2)}")
print()


# ==========================================
# 5) Fonction utilitaire : trouver chevauchement
# ==========================================
def find_overlapping_lesions_T2(label_T1):
    """
    Pour une lésion donnée à T1 (label_T1), on renvoie :
    - la liste des labels T2 qui chevauchent cette lésion
    - la taille (nombre de voxels) du chevauchement pour chacun
    """
    # masque binaire de cette lésion à T1
    lesion_T1 = sitk.Equal(cc_T1, label_T1)
    lesion_T1 = sitk.Cast(lesion_T1, sitk.sitkUInt8)

    # On "masque" cc_T2 par cette lésion :
    # Là où lesion_T1 = 0 -> 0
    # Là où lesion_T1 = 1 -> cc_T2 garde ses labels
    overlap_img = sitk.Mask(cc_T2, lesion_T1)

    overlap_stats = sitk.LabelShapeStatisticsImageFilter()
    overlap_stats.Execute(overlap_img)

    overlap_labels = list(overlap_stats.GetLabels())  # labels T2 qui chevauchent
    overlap_labels.sort()

    # On construit un dict label_T2 -> nb_voxels de chevauchement
    result = {}
    for lab2 in overlap_labels:
        num_pix = overlap_stats.GetNumberOfPixels(lab2)
        result[lab2] = num_pix

    return result


# ==================================================
# 6) Mise en correspondance et classification T1->T2
# ==================================================
matched_T2_labels = set()

# seuil de ratio de volume pour considérer lésion stable
# ex : si volume_T2 entre 0.8 et 1.2 * volume_T1 -> stable
ratio_low  = 0.8
ratio_high = 1.2

for lab1 in labels_T1:
    vol1_vox = stats_T1.GetNumberOfPixels(lab1)
    vol1_mm3 = stats_T1.GetPhysicalSize(lab1)
    centroid1 = stats_T1.GetCentroid(lab1)

    overlaps = find_overlapping_lesions_T2(lab1)

    if len(overlaps) == 0:
        print(f"Lésion T1 #{lab1} :")
        print(f"  Volume T1 : {vol1_mm3:.1f} mm^3")
        print(f"  Centroid T1 : {centroid1}")
        print(f"  → Aucune lésion correspondante à T2 (lésion disparue / résolue).")
        print()
        continue

    # S'il y a plusieurs labels T2 qui chevauchent, on prend celui avec le + grand recouvrement
    best_lab2 = None
    best_overlap = -1
    for lab2, nvox in overlaps.items():
        if nvox > best_overlap:
            best_overlap = nvox
            best_lab2 = lab2

    matched_T2_labels.add(best_lab2)

    vol2_vox = stats_T2.GetNumberOfPixels(best_lab2)
    vol2_mm3 = stats_T2.GetPhysicalSize(best_lab2)
    centroid2 = stats_T2.GetCentroid(best_lab2)

    ratio = vol2_mm3 / (vol1_mm3 + 1e-6)

    # Classification simple en 3 types + 1 cas
    if ratio_low <= ratio <= ratio_high:
        lesion_type = "Type 1 : lésion stable (volume similaire entre T1 et T2)"
    elif ratio > ratio_high:
        lesion_type = "Type 2 : lésion en progression (augmentation de volume)"
    else:  # ratio < ratio_low
        lesion_type = "Type 3 : lésion en régression (diminution de volume)"

    print(f"Lésion T1 #{lab1} ↔ Lésion T2 #{best_lab2}")
    print(f"  Volume T1 : {vol1_mm3:.1f} mm^3 ({vol1_vox} voxels)")
    print(f"  Volume T2 : {vol2_mm3:.1f} mm^3 ({vol2_vox} voxels)")
    print(f"  Ratio     : {ratio:.2f}")
    print(f"  Centroid T1 : {centroid1}")
    print(f"  Centroid T2 : {centroid2}")
    print(f"  Classification : {lesion_type}")
    print()


# ==========================================
# 7) Lésions nouvelles à T2 (non vues à T1)
# ==========================================
new_lesions_T2 = [lab2 for lab2 in labels_T2 if lab2 not in matched_T2_labels]

if len(new_lesions_T2) > 0:
    print("Lésions NOUVELLES à T2 (pas de correspondance à T1) :")
    for lab2 in new_lesions_T2:
        vol2_vox = stats_T2.GetNumberOfPixels(lab2)
        vol2_mm3 = stats_T2.GetPhysicalSize(lab2)
        centroid2 = stats_T2.GetCentroid(lab2)
        print(f"  - Lésion T2 #{lab2} : {vol2_mm3:.1f} mm^3, centroid {centroid2}")
else:
    print("Aucune lésion nouvelle à T2.")
