# AerialMetric-Oblique

This folder contains the oblique-view training assets for AerialMetric.

## Contents

1. `B` and `B-2` are two different sparse LiDAR depth datasets.
   - `B` comes from `UAVScenes`.
   - `B-2` comes from `GauUScene`.
2. `FOV` contains two subsets with unusual field-of-view distributions across the full dataset.
   - `PolyTech`
   - `ArtSci`
3. `TrainingData_Final_MoGe-All` contains dense depth pairs produced by projecting reconstructed meshes in AerialMetric.

## Training Note

Different depth types are assigned different weights during training, depending on the supervision source and density.
