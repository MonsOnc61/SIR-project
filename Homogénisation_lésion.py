import SimpleITK as sitk

# ======================
# 1) Chemins
# ======================
mask_path = "P1_T1_MASK.nii.gz"              # masque binaire des lésions (0/1)
output_path = "P1_T1_LESIONS_SORTED.nii.gz"  # sortie : labels normalisés

# ======================
# 2) Lecture du masque
# ======================
mask_img = sitk.ReadImage(mask_path)

# Sécurité : forcer binaire
mask_bin = sitk.BinaryThreshold(
    mask_img,
    lowerThreshold=0.5,
    upperThreshold=1e9,
    insideValue=1,
    outsideValue=0
)
mask_bin = sitk.Cast(mask_bin, sitk.sitkUInt8)

# ======================
# 3) Composantes connexes
# ======================
cc_img = sitk.ConnectedComponent(mask_bin)

# ======================
# 4) Calcul des volumes
# ======================
shape_stats = sitk.LabelShapeStatisticsImageFilter()
shape_stats.Execute(cc_img)

labels = list(shape_stats.GetLabels())

if len(labels) == 0:
    raise RuntimeError("Aucune lésion détectée dans le masque.")

# Associer label -> volume
lesions = []
for lab in labels:
    volume = shape_stats.GetPhysicalSize(lab)  # mm³
    lesions.append((lab, volume))

# ======================
# 5) Tri par volume décroissant
# ======================
lesions_sorted = sorted(lesions, key=lambda x: x[1], reverse=True)

print("Ordre des lésions (volume décroissant) :")
for new_id, (old_lab, vol) in enumerate(lesions_sorted, start=1):
    print(f"Lésion {new_id} ← ancien label {old_lab} | volume = {vol:.1f} mm³")

# ======================
# 6) Création de la nouvelle image de labels
# ======================
sorted_img = sitk.Image(cc_img.GetSize(), sitk.sitkUInt16)
sorted_img.CopyInformation(cc_img)

for new_id, (old_lab, _) in enumerate(lesions_sorted, start=1):
    mask_old = sitk.Equal(cc_img, old_lab)
    sorted_img = sitk.Add(
        sorted_img,
        sitk.Cast(mask_old, sitk.sitkUInt16) * new_id
    )

# ======================
# 7) Sauvegarde
# ======================
sitk.WriteImage(sorted_img, output_path)

print("\nFichier généré :", output_path)

