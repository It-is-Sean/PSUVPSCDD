# SCRREAM WAN Probe Final Summary — 2026-05-19

This document freezes the 2026-05 WAN Route2 audit state before moving to the next backbone. WAN is paused as a main branch, not deleted: the code, caches, Slurm scripts, and result folders remain useful references for future appendix checks.

## Fixed Probe Protocol

All WAN comparisons used the same SCRREAM / NOVA setup as the clean-GT VGGT ablation:

- data: `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
- split: `train=317`, `val=12`
- GT: sequence-meta-filtered registered SCRREAM mesh-complete targets, `20000` points, first input camera frame
- decoder / loss: frozen NOVA `scene_ae`, `nova_flow`
- validation: full robust val split, `val_visual_40960`, `best_fscore_010.pth`, `best_pred_to_gt_p90.pth`
- WAN T2V checkpoint: `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers`
- WAN T2V cache: `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81`

The main T2V hidden cache uses 81-frame video context around each SCRREAM pair and saves pair temporal slices as `[3120,1536]`.

## Primary Results

| branch | representative setting | F@0.10 | pred-to-GT p90 | Chamfer-L2 | conclusion |
| --- | --- | ---: | ---: | ---: | --- |
| Clean-GT VGGT reference | VGGT layer16 | `0.6860468604251301` | `0.20770130679011345` | `0.026904070439438026` | much stronger than WAN |
| Old WAN hidden MLP | T2V `ctx81 t499/layer09` | `0.46988987902779306` | `0.48718947172164917` | `0.21040735269586244` | live baseline above controls |
| Hidden CA historical peak | T2V `ctx81 t499/layer14` | `0.5012956284974035` | `0.4229188362757365` | `0.10908368105689685` | best WAN metric observed |
| Hidden CA repeated F baseline | T2V `ctx81 t249/layer14` | `0.48913902331806663` | `0.4372795696059863` | `0.12182091859479745` | strongest repeated F setting |
| Hidden CA repeated geometry baseline | T2V `ctx81 t249/layer09` | `0.4766100401399189` | `0.410525918006897` | `0.11981592203179996` | best repeated p90 / Chamfer |
| Hidden CA L4 capacity | T2V `ctx81 t249/layer14`, `adapter_layers=4` | `0.4919946248489035` | `0.4724609777331352` | `0.11948200377325217` | small F gain, worse p90 |

## Negative Branches

| branch | representative result | conclusion |
| --- | --- | --- |
| Route2.1 no-noise / low-noise | best `no_noise/layer29`, F@0.10 `0.44840173` | clean / low-noise latent state did not fix WAN |
| Token normalization | `t499/layer09 + token_layernorm`, F@0.10 `0.45476213575933216` | feature scale was not the bottleneck |
| Pair tiled context | `pair_tiled81 t499/layer09`, F@0.10 `0.4429200036613287` | real 81-frame context was not the main failure |
| Pred-x0 latent Conv2d | `pred_x0_latent t499`, F@0.10 `0.41336424426176693` | denoised latent is weak as NOVA condition |
| Model-output latent Conv2d | `model_output_latent t499`, F@0.10 `0.4199299775478907` | raw model output latent is weak |
| Latent CA readout | best `pred_x0_latent + latent CA`, F@0.10 `0.43766060222664477` | learned latent readout still below hidden CA |
| Fixed hidden grid readouts | `grid2d_pool` F@0.10 `0.43326753863230877`; `grid2d_conv` `0.39855373112050535` | fixed 2D alignment is insufficient |
| Multi-layer hidden fusion | `t249 layers 9+14+19`, F@0.10 `0.4724058254433277` | simple same-timestep fusion did not help |
| Multi-timestep hidden fusion | `249+499+749/layer14`, F@0.10 `0.45213117002164466` | simple same-layer timestep fusion did not help |
| FLF2V hidden | `pair_endpoint81 t249/layer14`, F@0.10 `0.42897984457682026` | first/last-frame 14B hidden did not improve |
| Gated hidden CA | best `t499/layer14`, F@0.10 `0.4866296484297818` | no reliable gain over standard hidden CA |

## Interpretation

WAN should be reported as a setting-sensitive but informative probe:

- WAN T2V hidden states contain usable spatial signal: real features beat zero/sample-shuffle controls, and learned CA readout improves over the MLP hidden baseline.
- The WAN-to-NOVA interface is the main positive lesson. `wan_cross_attn_resampler` is the only readout family that consistently made WAN competitive with the old hidden baseline.
- WAN still trails clean-GT VGGT by a large margin under the same SCRREAM / NOVA decoder protocol. The best WAN historical F@0.10 is about `0.50`; repeated settings are around `0.49`; VGGT layer16 is `0.6860`.
- Simple explanations were ruled out: diffusion noise level, token normalization, context dilution, latent tensor choice, simple grid pooling, simple hidden fusion, FLF2V hidden, and gated residual CA did not close the gap.
- The `t499/layer14` CA result should be treated as a historical peak, not a stable mean. `t249/layer09` and `t249/layer14` are the safer repeated comparison anchors.

## Final WAN Policy

WAN is paused as of `2026-05-19`.

Do not spend more broad sweeps on:

- WAN latent tensors (`pred_x0_latent`, `model_output_latent`)
- simple multi-layer / multi-timestep hidden fusion
- FLF2V hidden expansion
- gated CA expansion
- late single layers such as layer24 / layer29

Optional future WAN appendix checks, only if needed:

1. visually compare full-grid `t249/layer09`, full-grid `t249/layer14`, and L4 `t249/layer14`;
2. run L4 hidden CA at `t249/layer09`;
3. repeat L4 hidden CA at `t249/layer14` with a second seed.

## Lessons For The Next Backbone

The next model should reuse the clean SCRREAM / NOVA probe protocol but avoid the WAN sweep explosion:

1. Start with two or three representative layers.
2. Preserve native grid metadata in the cache.
3. Run a zero or sample-shuffle sanity control early.
4. Smoke one training step before full runs.
5. Try the model-native token projection first, but move quickly to a learned cross-attention resampler if fixed pooling is weak.
6. Only expand layer/timestep grids after the readout interface is validated.
