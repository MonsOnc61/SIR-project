import SimpleITK as sitk

binary_path = "binary_volume.nii.gz"

img = sitk.ReadImage(binary_path)

binary = sitk.Cast(sitk.NotEqual(img, 0), sitk.sitkUInt8)

cc_filter = sitk.ConnectedComponentImageFilter()
cc = cc_filter.Execute(binary)

stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

labels = stats.GetLabels()

if len(labels) == 0:
    print("Aucune partie blanche détectée.")
    exit()

print("===================================")
print("        ANALYSE DU VOLUME")
print("===================================")
print(f"Nombre de parties blanches : {len(labels)}")

total_volume = sum(stats.GetPhysicalSize(lb) for lb in labels)
print(f"Volume total (mm^3) : {total_volume:.2f}")
print()

for lb in labels:
    vol = stats.GetPhysicalSize(lb)
    centroid = stats.GetCentroid(lb)
    bbox = stats.GetBoundingBox(lb)

    print(f"--- Partie {lb} ---")
    print(f"Volume :    {vol:.2f} mm^3")
    print(f"Centre :    {centroid}")
    print(f"BBox (idx,size) : {bbox}")
    print()
