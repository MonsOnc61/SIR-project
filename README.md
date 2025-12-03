# SIR-project

Projet SIR — Segmentation des Lésions SEP en IRM


Contexte

La sclérose en plaques (SEP) est une maladie neurologique chronique caractérisée par l’apparition de lésions démyélinisantes dans le système nerveux central.
Ces lésions sont visibles en IRM, notamment sur la séquence FLAIR, et leur segmentation est essentielle pour :

le diagnostic initial,

le suivi de la progression de la maladie,

l’évaluation des traitements.

La segmentation manuelle reste cependant :

longue et fastidieuse,

dépendante de l’expertise du clinicien,

fortement variable d’un annotateur à l’autre.

Pour répondre à ces limites, ce projet s’appuie sur le dataset MSLesSeg, un corpus de référence fournissant :

IRM multimodales (T1, T2, FLAIR),

annotations expertes,

pipeline de pré-traitements standardisés.

Ce dataset est conçu pour accompagner la recherche en segmentation automatique des lésions SEP via l'IA.


Objectifs du projet

Le projet SIR vise à offrir une introduction complète à un pipeline réel en imagerie médicale en abordant :

la prise en main du dataset MSLesSeg,

l’étude des pré-traitements (BET, FLIRT),

l’exploration et la visualisation des segmentations,

la manipulation d’images 3D via SimpleITK,

l’analyse de méthodes de segmentation IA basées sur l’article MSLesSeg,

la reproduction d’un modèle baseline.



Contenu du projet

Code Python sous Jupyter Notebook

Scripts de pré-traitements (BET, FLIRT, SimpleITK)

Exploration des volumes IRM et des annotations

Pipeline IA (loader PyTorch, normalisation, cropping, entraînement)

Analyse quantitative et qualitative

Rapport final + présentation
