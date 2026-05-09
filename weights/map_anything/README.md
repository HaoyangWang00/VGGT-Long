---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- computer-vision
- 3d-reconstruction
- multi-view-stereo
- depth-estimation
- camera-pose
- covisibility
- mapanything
license: cc-by-nc-4.0
language:
- en
pipeline_tag: image-to-3d
---

## Overview

MapAnything is a simple, end-to-end trained transformer model that directly regresses the factored metric 3D geometry of a scene given various types of modalities as inputs. A single feed-forward model supports over 12 different 3D reconstruction tasks including multi-image sfm, multi-view stereo, monocular metric depth estimation, registration, depth completion and more.

This is the CC-BY-NC-4.0 variant of the model. Latest release on Jan 20th 2026.

## Quick Start

Please refer to our Github Repo: https://github.com/facebookresearch/map-anything

## Citation

If you find our repository useful, please consider giving it a star ⭐ and citing our paper in your work:

```bibtex
@inproceedings{keetha2026mapanything,
  title={{MapAnything}: Universal Feed-Forward Metric 3D Reconstruction},
  author={Keetha, Nikhil and M{\"u}ller, Norman and Sch{\"o}nberger, Johannes and Porzi, Lorenzo and Zhang, Yuchen and Fischer, Tobias and Knapitsch, Arno and Zauss, Duncan and Weber, Ethan and Antunes, Nelson and others},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2026},
  organization={IEEE}
}
```