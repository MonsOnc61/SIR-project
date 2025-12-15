import SimpleITK as sitk

# Chemins à adapter
aseg_path = "aseg.nii.gz"
t1_path   = "T1.nii.gz"

# Lecture des images
aseg_img = sitk.ReadImage(aseg_path)
t1_img   = sitk.ReadImage(t1_path)  # pas utilisé pour l'instant mais OK

# LUT FreeSurfer ASEG
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
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",

    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex", 
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
}

# LUT APARC (utile plus tard)
freesurfer_aparc_labels = {
    1000: "lh-Unknown",
    1001: "lh-Bankssts",
    1002: "lh-Caudal-Middle-Frontal",
    1003: "lh-Caudal-Anterior-Cingulate",
    1005: "lh-Cuneus",
    1006: "lh-Entorhinal",
    1007: "lh-Fusiform",
    1008: "lh-Inferior-Parietal",
    1009: "lh-Inferior-Temporal",
    1010: "lh-Isthmus-Cingulate",
    1011: "lh-Lateral-Occipital",
    1012: "lh-Lateral-Orbitofrontal",
    1013: "lh-Lingual",
    1014: "lh-Medial-Orbitofrontal",
    1015: "lh-Middle-Temporal",
    1016: "lh-Parahippocampal",
    1017: "lh-Paracentral",
    1018: "lh-Pars-Opercularis",
    1019: "lh-Precentral",
    1020: "lh-Precuneus",
    1021: "lh-Rostral-Anterior-Cingulate",
    1022: "lh-Rostral-Middle-Frontal",
    1023: "lh-Superior-Frontal",
    1024: "lh-Superior-Parietal",
    1025: "lh-Superior-Temporal",
    1026: "lh-Supramarginal",
    1027: "lh-Frontal-Pole",
    1028: "lh-Temporal-Pole",
    1029: "lh-Transverse-Temporal",

    # Hémisphère droit
    2000: "rh-Unknown",
    2001: "rh-Bankssts",
    2002: "rh-Caudal-Middle-Frontal",
    2003: "rh-Caudal-Anterior-Cingulate",
    2005: "rh-Cuneus",
    2006: "rh-Entorhinal",
    2007: "rh-Fusiform",
    2008: "rh-Inferior-Parietal",
    2009: "rh-Inferior-Temporal",
    2010: "rh-Isthmus-Cingulate",
    2011: "rh-Lateral-Occipital",
    2012: "rh-Lateral-Orbitofrontal",
    2013: "rh-Lingual",
    2014: "rh-Medial-Orbitofrontal",
    2015: "rh-Middle-Temporal",
    2016: "rh-Parahippocampal",
    2017: "rh-Paracentral",
    2018: "rh-Pars-Opercularis",
    2019: "rh-Precentral",
    2020: "rh-Precuneus",
    2021: "rh-Rostral-Anterior-Cingulate",
    2022: "rh-Rostral-Middle-Frontal",
    2023: "rh-Superior-Frontal",
    2024: "rh-Superior-Parietal",
    2025: "rh-Superior-Temporal",
    2026: "rh-Supramarginal",
    2027: "rh-Frontal-Pole",
    2028: "rh-Temporal-Pole",
    2029: "rh-Transverse-Temporal",
}

def get_structure_name(label, parcellation="aseg"):
    if parcellation == "aseg":
        return freesurfer_aseg_labels.get(label, f"Label {label} (inconnu dans ASEG)")
    elif parcellation == "aparc":
        return freesurfer_aparc_labels.get(label, f"Label {label} (inconnu dans APARC)")
    return f"Label {label}"

# Choisir la structure à analyser (exemple : pallidum gauche = 13)
label_value = 26

# Création du masque binaire SANS NumPy
roi_img = sitk.Equal(aseg_img, label_value)
roi_img = sitk.Cast(roi_img, sitk.sitkUInt8)  # image binaire propre

# Analyse morphologique
label_stats = sitk.LabelShapeStatisticsImageFilter()
label_stats.Execute(roi_img)

if not label_stats.HasLabel(1):
    raise ValueError("La structure est vide dans ce sujet.")

volume_mm3   = label_stats.GetPhysicalSize(1)
centroid_mm  = label_stats.GetCentroid(1)
bbox         = label_stats.GetBoundingBox(1)
elongation   = label_stats.GetElongation(1)
flatness     = label_stats.GetFlatness(1)

structure_name = get_structure_name(label_value, parcellation="aseg")

print("=== Résultats ASEG ===")
print(f"Label: {label_value}")
print(f"Structure: {structure_name}")
print(f"Volume: {volume_mm3:.1f} mm^3")
print(f"Centroid (mm): {centroid_mm}")
print(f"Bounding box (index, size): {bbox}")
print(f"Elongation: {elongation:.3f}")
print(f"Flatness: {flatness:.3f}")
