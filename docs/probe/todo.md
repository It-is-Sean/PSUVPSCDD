# TODO

## Repo / infra

- [x] fold the proposal workspace into the integrated `PSUVPSC3DD_repo`
- [x] vendor the minimum external code needed for the first loop (`third_party/vggt`, `dust3r/datasets`, `datasets_preprocess`)
- [x] scaffold docs / configs / probe package / launcher scripts
- [ ] decide whether this repo will remain local-first or be mirrored to a cleaner remote research fork later

## Canonical decoder

- [ ] inspect Stage-1 checkpoint format and latent shape
- [ ] document how to call the frozen decoder independently of the original training stack
- [ ] verify the exact flow-matching inference schedule used by Stage 1

## Feature extraction

- [ ] define a standard frozen-representation tensor contract
- [ ] implement NOVA3R extractor first
- [ ] add at least one non-NOVA3R geometry backbone
- [ ] add the first video backbone

## Probing

- [ ] replace launcher stubs with real train loops
- [ ] implement direct baseline A
- [ ] implement visible / unseen split metrics
- [ ] export run cards for every serious experiment

## Paper-facing analysis

- [ ] narrow the **first paper scope**: image/geometry only vs image+video
- [ ] lock the main claim / experiment matrix
- [ ] define main-table columns before running large sweeps
- [ ] decide which backbones belong in the first serious matrix versus later extensions
- [ ] decide which ablations stay in the main paper vs appendix
