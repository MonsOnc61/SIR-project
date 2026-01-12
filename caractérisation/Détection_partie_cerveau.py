import SimpleITK as sitk
import csv

# ==========================
# Chemins à adapter
# ==========================
aseg_path = "aseg.nii.gz"
t1_path   = "T1.nii.gz"

# ==========================
# Lecture des images
# ==========================
aseg_img = sitk.ReadImage(aseg_path)
t1_img   = sitk.ReadImage(t1_path)  # pas utilisé pour l'instant

# ==========================
# LUT FreeSurfer ASEG
# ==========================
freesurfer_aseg_labels = {
    0:  "Background",
    2:  "Left-Cerebral-White-Matter",
    3:  "Left-Cerebral-Cortex",
    4:  "Left-Lateral-Ventricle",
    5:  "Left-Inf-Lat-Vent",
    7:  "Left-Cerebellum-White-Matter",
    8:  "Left-Cerebellum-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    26: "Left-Accumbens-area",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area"
}

def get_structure_name(label_value, parcellation="aseg"):
    if parcellation == "aseg":
        return freesurfer_aseg_labels.get(label_value, "Unknown")
    return "Unknown"

# ==========================
# Sélection du label étudié
# ==========================
label_value = 17  # exemple : hippocampe gauche

binary = sitk.Equal(aseg_img, label_value)

label_stats = sitk.LabelShapeStatisticsImageFilter()
label_stats.Execute(binary)

# ==========================
# Mesures
# ==========================
volume_mm3  = label_stats.GetPhysicalSize(1)
centroid_mm = label_stats.GetCentroid(1)
bbox        = label_stats.GetBoundingBox(1)
elongation  = label_stats.GetElongation(1)
flatness    = label_stats.GetFlatness(1)

structure_name = get_structure_name(label_value, parcellation="aseg")

# ==========================
# PRINT (inchangé)
# ==========================
print("=== Résultats ASEG ===")
print(f"Label: {label_value}")
print(f"Structure: {structure_name}")
print(f"Volume: {volume_mm3:.1f} mm^3")
print(f"Centroid (mm): {centroid_mm}")
print(f"Bounding box (index, size): {bbox}")
print(f"Elongation: {elongation:.3f}")
print(f"Flatness: {flatness:.3f}")

# ==========================
# CSV – écriture des mêmes infos
# ==========================
csv_path = "resultats_aseg.csv"

with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)

    # En-tête
    writer.writerow([
        "label",
        "structure",
        "volume_mm3",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "bbox_index_x",
        "bbox_index_y",
        "bbox_index_z",
        "bbox_size_x",
        "bbox_size_y",
        "bbox_size_z",
        "elongation",
        "flatness"
    ])

    # Données (copie exacte de ce qui est print)
    writer.writerow([
        label_value,
        structure_name,
        volume_mm3,
        centroid_mm[0],
        centroid_mm[1],
        centroid_mm[2],
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        bbox[4],
        bbox[5],
        elongation,
        flatness
    ])

print(f"\nCSV généré : {csv_path}")
