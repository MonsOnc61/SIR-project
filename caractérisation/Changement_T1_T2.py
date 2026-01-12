import SimpleITK as sitk
import math
import csv

# ======================
# 1) Chemins
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

# ==========================
# 3) Connected components
# ==========================
cc_T1 = sitk.ConnectedComponent(mask_T1_img)
cc_T2 = sitk.ConnectedComponent(mask_T2_img)

stats_T1 = sitk.LabelShapeStatisticsImageFilter()
stats_T2 = sitk.LabelShapeStatisticsImageFilter()

stats_T1.Execute(cc_T1)
stats_T2.Execute(cc_T2)

labels_T1 = stats_T1.GetLabels()
labels_T2 = stats_T2.GetLabels()

# ==========================
# 4) CSV – ouverture
# ==========================
csv_path = "resultats_lesions_T1_T2.csv"
csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "temps",
    "label",
    "volume_voxels",
    "volume_mm3",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "type_lesion"
])

# ==========================
# 5) Appariement T1 → T2
# ==========================
distance_threshold = 10.0  # mm
matched_T2_labels = set()

print("\nLésions persistantes (T1 → T2) :")

for lab1 in labels_T1:
    centroid1 = stats_T1.GetCentroid(lab1)
    vol1_vox = stats_T1.GetNumberOfPixels(lab1)
    vol1_mm3 = stats_T1.GetPhysicalSize(lab1)

    best_lab2 = None
    best_dist = float("inf")

    for lab2 in labels_T2:
        centroid2 = stats_T2.GetCentroid(lab2)
        dist = math.dist(centroid1, centroid2)

        if dist < best_dist:
            best_dist = dist
            best_lab2 = lab2

    if best_dist < distance_threshold:
        matched_T2_labels.add(best_lab2)

        vol2_vox = stats_T2.GetNumberOfPixels(best_lab2)
        vol2_mm3 = stats_T2.GetPhysicalSize(best_lab2)
        centroid2 = stats_T2.GetCentroid(best_lab2)

        print(f"- T1 #{lab1} → T2 #{best_lab2} | {vol1_mm3:.1f} → {vol2_mm3:.1f} mm³")

        # CSV T1
        csv_writer.writerow([
            "T1",
            lab1,
            vol1_vox,
            vol1_mm3,
            centroid1[0],
            centroid1[1],
            centroid1[2],
            "persistante"
        ])

        # CSV T2
        csv_writer.writerow([
            "T2",
            best_lab2,
            vol2_vox,
            vol2_mm3,
            centroid2[0],
            centroid2[1],
            centroid2[2],
            "persistante"
        ])

# ==========================================
# 6) Lésions nouvelles à T2
# ==========================================
new_lesions_T2 = [lab2 for lab2 in labels_T2 if lab2 not in matched_T2_labels]

if len(new_lesions_T2) > 0:
    print("\nLésions NOUVELLES à T2 :")
    for lab2 in new_lesions_T2:
        vol2_vox = stats_T2.GetNumberOfPixels(lab2)
        vol2_mm3 = stats_T2.GetPhysicalSize(lab2)
        centroid2 = stats_T2.GetCentroid(lab2)

        print(f"  - T2 #{lab2} : {vol2_mm3:.1f} mm³, centroid {centroid2}")

        csv_writer.writerow([
            "T2",
            lab2,
            vol2_vox,
            vol2_mm3,
            centroid2[0],
            centroid2[1],
            centroid2[2],
            "nouvelle"
        ])
else:
    print("\nAucune lésion nouvelle à T2.")

# ==========================
# 7) Fermeture CSV
# ==========================
csv_file.close()
print(f"\nCSV généré : {csv_path}")
