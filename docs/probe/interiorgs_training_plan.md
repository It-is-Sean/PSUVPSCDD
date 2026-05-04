# InteriorGS training migration plan

Status as of 2026-05-03: deferred. This was the next planned data direction
after the 2026-04-29 ScanNet diagnosis, but the immediate branch has moved to
full SCRREAM mesh-complete adapter training because the full SCRREAM tree is now
available at `~/datasets/SCRREAM`.

## Motivation

The current ScanNet line is still useful as a debugging and baseline track. This
InteriorGS plan remains useful if the project returns to a higher-quality indoor
dataset after the corrected SCRREAM baseline is understood.

InteriorGS is the current candidate dataset:

- upstream repo: `https://github.com/manycore-research/InteriorGS`
- dataset page: `https://huggingface.co/datasets/spatialverse/InteriorGS`
- current release notes, as of 2026-04-29: v2.0 lists 1,000 indoor scenes and
  floorplans.
- per-scene assets include `3dgs_compressed.ply`, `labels.json`,
  `occupancy.png`, `occupancy.json`, and `structure.json`.
- upstream docs describe more than 554k object instances across 755 categories,
  with oriented 3D boxes.
- upstream docs define the 3DGS coordinate system as XYZ = Right, Back, Up, with
  units in meters.

## Why it fits this project

InteriorGS is not a drop-in replacement for the existing ScanNet mesh pipeline.
It is a 3D Gaussian Splatting dataset, not a ScanNet-style RGB-D mesh dataset.
That makes it attractive as a high-quality supervision source, but it requires a
new adapter/data bridge before training.

Useful properties for the PSUVPSC3DD direction:

- dense, high-quality indoor 3D scene representation;
- object-level semantic labels and 3D boxes;
- occupancy maps and floorplan/layout metadata;
- coordinate system and metric units documented by the dataset;
- scene count is large enough for a real training split instead of a small
  smoke-test-only branch.

## Local server plan

1. Accept the dataset terms on Hugging Face and download a small pilot subset on
   this server first.
2. Inspect one to three scenes directly:
   - parse `3dgs_compressed.ply`;
   - parse `labels.json`;
   - parse `structure.json`;
   - verify coordinate frame, units, bounds, and object boxes.
3. Decide the first supervision conversion:
   - render multi-view RGB/depth from 3DGS and feed the existing VGGT/NOVA path;
   - sample point clouds from 3DGS centers/covariances as an approximate target;
   - or use layout/object boxes as an auxiliary target while keeping point
     reconstruction as the main probe.
4. Add a minimal `InteriorGSDataset` path under the probe workspace instead of
   bending the existing ScanNet loader too far.
5. Start with a tiny split and a fixed evaluation manifest, then scale only
   after the data bridge produces sane renders and target point clouds.

## First implementation checkpoint

The first useful checkpoint is not full training. It is a data sanity artifact:

- a script that reads a small InteriorGS scene subset;
- a stable train/val/test split file;
- one rendered or sampled target point cloud per scene;
- one visual inspection sheet showing input views, target points, occupancy, and
  semantic boxes;
- documented coordinate transform into the current NOVA/VGGT convention.

Only after that should the project launch adapter training on InteriorGS.
