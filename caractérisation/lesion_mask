import SimpleITK as sitk

binary_path = "binary_volume.nii.gz"

img = sitk.ReadImage(binary_path)

binary = sitk.Cast(sitk.NotEqual(img, 0), sitk.sitkUInt8)

cc_filter = sitk.ConnectedComponentImageFilter()
cc = cc_filter.Execute(binary)

stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(cc)

labels = list(range(1, cc_filter.GetObjectCount() + 1))

if len(labels) == 0:
    print("Aucune partie blanche détectée.")
    exit()

def classify_shape(principal_axes):
    """
    Déduit la forme à partir des axes principaux (longueurs A >= B >= C).
    """
    A, B, C = principal_axes

    ratio_AB = A / B if B > 0 else float("inf")
    ratio_AC = A / C if C > 0 else float("inf")
    ratio_BC = B / C if C > 0 else float("inf")

    # Critères
    if ratio_AB < 1.5 and ratio_AC < 1.5:
        return "sphérique"

    if ratio_AC > 2.5 and ratio_AB > 1.5:
        return "allongée (en tige)"

    if ratio_BC > 2.5 and ratio_AB < 1.2:
        return "plate (compressée)"

    return "irrégulière"

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
    principal_axes = stats.GetPrincipalMoments(lb)
    shape = classify_shape(principal_axes)
    print(f"Forme       : {shape}")
    print()
