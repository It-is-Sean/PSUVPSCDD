# TODO

## Repo / infra

- [x] clone upstream NOVA3R under `projects/`
- [x] create proposal branch
- [x] scaffold docs / configs / probe package / launcher scripts
- [ ] decide whether to keep this as a pure local branch or mirror to a remote fork later

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

- [ ] lock the main claim / experiment matrix
- [ ] define main-table columns before running large sweeps
- [ ] decide which ablations stay in the main paper vs appendix
