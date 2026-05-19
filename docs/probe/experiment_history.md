# Experiment history summary

## 2026-05-19 WAN final pause

WAN Route2 is paused as a main branch after the 2026-05-19 audit. The final consolidated summary is `docs/probe/wan_summary_2026-05-19.md`.

Frozen conclusion:

- WAN hidden states contain usable spatial signal, but they require a learned CA resampler to align with NOVA/FM condition tokens.
- Historical best WAN is `t499/layer14 + wan_cross_attn_resampler`, F@0.10 `0.5012956284974035`, while repeated settings are closer to `0.49`.
- WAN remains far below clean-GT VGGT layer16, F@0.10 `0.6860468604251301`.
- Noise/noise-free settings, token normalization, pair_tiled81, latent tensors, fixed grid readouts, multi-source fusion, FLF2V hidden, gated CA, and L4 capacity did not justify further broad WAN sweeps.
- Future WAN work is appendix-only; the active model-coverage direction moves to the next backbone under the same SCRREAM / NOVA protocol.

## 2026-05-19 WAN hidden CA stability/capacity completion

The hidden CA stability/capacity branch completed after the latent tensor-choice and FLF2V audits. All jobs exited `0:0` on `air-node-02`.

| job | setting | F@0.10 | pred-to-GT p90 | Chamfer-L2 | interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| `86547` | standard hidden CA seed23, `t249/layer14` | `0.48706357506233194` | `0.4482837840914726` | `0.11202508273224036` | close to full-grid `t249/layer14`, confirms variance |
| `86548` | standard hidden CA seed23, `t499/layer14` | `0.4591392560862819` | `0.4479780395825704` | `0.14107579924166203` | below historical `t499/layer14` peak |
| `86549` | gated hidden CA seed17, `t249/layer14` | `0.4505522726705196` | `0.46173084527254105` | `0.11595033543805282` | gated residual CA negative by F-score |
| `86550` | gated hidden CA seed17, `t499/layer14` | `0.4866296484297818` | `0.45513606319824856` | `0.12565729891260466` | decent but below historical peak |
| `86553` | standard hidden CA L4/H512 seed17, `t249/layer14` | `0.4919946248489035` | `0.4724609777331352` | `0.11948200377325217` | best recent repeated-F result, small positive |
| `86554` | standard hidden CA L4/H512 seed17, `t499/layer14` | `0.46280321302050603` | `0.44610939423243207` | `0.13269949393967786` | negative versus historical `t499/layer14` |

Interpretation:

- hidden CA remains the only clearly positive WAN readout family;
- L4 `t249/layer14` slightly improves over the full-grid repeated `t249/layer14` F-score (`0.49199` vs `0.48914`), but it does not beat the historical `t499/layer14` peak and its p90 is worse than full-grid `t249/layer09`;
- gated CA does not justify expansion;
- seed variance is large enough that `t499/layer14=0.5013` should still be treated as a historical peak, not a stable mean.

Recommended next WAN-only checks:

1. visually inspect full-grid `t249/layer09`, full-grid `t249/layer14`, and L4 `t249/layer14`;
2. if continuing this branch, run L4 `t249/layer09` and one L4 `t249/layer14` seed repeat;
3. do not expand latent tensors, simple fusion, FLF2V hidden, gated CA, or weak late layers unless visuals contradict the metrics.

## 2026-05-19 WAN latent audit and hidden CA stability follow-up

The previous hidden-fusion branch finished negative, and the first FLF2V hidden run also completed. T2V latent tensor-choice audits were then completed with both fixed Conv2d and learned latent-grid CA readouts. They did not beat the hidden CA baselines, so the WAN-only search moved back to the hidden CA-resampler branch. The stability/capacity jobs described in this section later completed; see the section above.

Completed results:

| job | setting | F@0.10 | pred-to-GT p90 | Chamfer-L2 | interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| `86494` | T2V CA multi-timestep `249+499+749/layer14` | `0.45213117002164466` | `0.49119475732247037` | `0.12659800921877226` | negative versus single-layer CA |
| `86520` | FLF2V hidden `pair_endpoint81 t249/layer14` | `0.42897984457682026` | `0.5051772321263949` | `0.14052311765650907` | live but weaker than T2V CA |
| `86536` | T2V `model_output_latent t499 + conv2d_mlp` | `0.4199299775478907` | `0.6285491685072581` | `0.44791099180777866` | negative versus hidden MLP and hidden CA |
| `86540` | T2V `model_output_latent t499 + latent CA resampler` | `0.4289946537162989` | `0.5942087918519974` | `0.19963246708114943` | better Chamfer than Conv2d, still negative versus hidden CA |
| `86543` | T2V `pred_x0_latent t499 + latent CA resampler` | `0.43766060222664477` | `0.5175856028993925` | `0.20880577837427458` | best latent learned-readout result, still below hidden CA |

FLF2V cache state:

- checkpoint retry/preflight job `86500` completed after the first `86497` incomplete-read failure;
- cache shard jobs `86515` / `86516` / `86517` / `86518` completed `329/329` samples under `experiments/probe3d/feature_cache/scrream_wan_flf2v14b_pair_endpoint81_480/t249/layer14`;
- feature shape is `[3120,5120]` at 480P, with `pair_endpoint81`, first/last-frame conditioning, and temporal indices `[0,20]`;
- smoke train job `86519` completed before the formal `86520`.

Active follow-up:

- `prepare_scrream_wan_t2v_feature_cache.py` now supports `--feature_kind model_output_latent`, caching raw WAN transformer `output.sample` latent slices as `[12480,16]`;
- `train_wan_t2v_nova_adapter.py` accepts `--wan_feature_kind model_output_latent`;
- `conv2d_mlp` is valid for `pred_x0_latent` and `model_output_latent`;
- `WanLatentCrossAttentionResamplerAdapter` / `--adapter_type wan_latent_cross_attn_resampler` is now implemented for latent feature kinds, restoring `[B,2,60,104,16]` and cross-attending NOVA query tokens to the full latent grid;
- `slurm/scrream_wan_t2v_precompute.sbatch` and `slurm/scrream_wan_t2v_ablation_pack_train.sbatch` have default cache/run prefixes for `model_output_latent`;
- Slurm chain `86534 -> 86535 -> 86536` completed the cache / one-step smoke / formal train path for `ctx81 normal t499 model_output_latent + conv2d_mlp`; the formal result was negative.
- Learned latent-readout follow-ups were started after the Conv2d result: `86539` validated `model_output_latent + wan_latent_cross_attn_resampler`, formal job `86540` completed negative, cancelled high-memory pred-x0 smoke `86541` was replaced by lower-memory smoke `86542`, and formal `pred_x0_latent + wan_latent_cross_attn_resampler` job `86543` completed negative versus hidden CA.

Follow-up submitted at that timestamp:

- `train_wan_t2v_nova_adapter.py` now supports `--adapter_gated` for zero-initialized residual gates in WAN CA blocks;
- `slurm/scrream_wan_t2v_ablation_pack_train.sbatch` now passes `WAN_ADAPTER_GATED` and `WAN_SEED`;
- static checks and local dummy forward/backward passed on `2026-05-19`;
- jobs `86547` / `86548` are standard hidden CA seed-23 repeats for `t249/layer14` and `t499/layer14`;
- jobs `86549` / `86550` are gated hidden CA seed-17 checks for `t249/layer14` and `t499/layer14`.

Interpretation at submission time: simple hidden-source fusion, FLF2V hidden, and latent tensor choices did not explain the WAN gap. The next useful question was whether the only positive branch, hidden CA-resampling, was stable enough to trust and whether gated residual CA could improve optimization. The completed result above says gated CA was not the answer, while L4 hidden CA is a small positive at `t249/layer14`.

## 2026-05-18 WAN multi-source hidden fusion audit

After the CA-resampler full-grid result, the next question was whether WAN geometry signal is distributed across multiple hidden sources rather than captured by a single timestep/layer. Two narrow fusion checks were added without regenerating caches:

- same timestep, multiple layers: `t249 layers 9+14` and `t249 layers 9+14+19`;
- same layer, multiple timesteps: `timesteps 249+499+749` at layer `14`.

Implementation:

- `WanHiddenMultiSourceCrossAttentionResamplerAdapter` restores concatenated hidden tokens as `[source, temporal, row, col]`, adds source/temporal/row/column position embeddings, and cross-attends learned NOVA query tokens to all WAN tokens.
- `train_wan_t2v_nova_adapter.py` accepts `--wan_layers` for multi-layer fusion and `--wan_timesteps` for multi-timestep fusion. Both modes are hidden-cache-only and cannot be combined in the same run.
- Slurm launchers were added:
  - `slurm/scrream_wan_t2v_multilayer_pack_train.sbatch`;
  - `slurm/scrream_wan_t2v_multitime_pack_train.sbatch`.
- `swanlab.finish()` is now guarded so a network/proxy error during teardown does not fail a completed training run after metrics are already written.

Multi-layer execution:

| job | setting | Slurm state | F@0.10 | pred-to-GT p90 | Chamfer-L2 | note |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| `86483` | smoke `t249 layers 9+14` | `COMPLETED 0:0` | n/a | n/a | n/a | wrote loss / robust val / preview artifacts |
| `86485` | `t249 layers 9+14` | `COMPLETED 0:0` | `0.46487238144059545` | `0.49916083614031476` | `0.12842474008599916` | negative versus single-layer CA |
| `86487` | `t249 layers 9+14+19` | `FAILED 1:0` | `0.4724058254433277` | `0.449202927450339` | `0.12805132629970709` | training completed; failure was only late `swanlab.finish()` proxy teardown |

Comparison baselines:

| setting | F@0.10 | pred-to-GT p90 | Chamfer-L2 |
| --- | ---: | ---: | ---: |
| single-layer CA full-grid `t249/layer14` | `0.48913902331806663` | `0.4372795696059863` | `0.12182091859479745` |
| single-layer CA full-grid `t249/layer09` | `0.4766100401399189` | `0.410525918006897` | `0.11981592203179996` |
| multi-layer `t249 layers 9+14` | `0.46487238144059545` | `0.49916083614031476` | `0.12842474008599916` |
| multi-layer `t249 layers 9+14+19` | `0.4724058254433277` | `0.449202927450339` | `0.12805132629970709` |

Interpretation:

- same-timestep multi-layer fusion did not beat single-layer CA-resampler baselines;
- `9+14+19` is slightly better than `9+14`, but still below `t249/layer14` on F-score and below `t249/layer09` on p90/Chamfer;
- stop simple multi-layer hidden fusion unless a later result suggests a more targeted layer combination.

Multi-timestep result, finalized on `2026-05-19`:

- smoke job `86493` completed successfully for `249+499+749/layer14`; the real input shape was `[1,9360,1536]`;
- formal job `86494` completed with `F@0.10=0.45213117002164466`, `pred-to-GT p90=0.49119475732247037`, and `Chamfer-L2=0.12659800921877226`.

Interpretation: same-layer multi-timestep hidden fusion did not improve over the single-layer CA baselines, so simple hidden-source fusion is stopped for now.

## 2026-05-18 WAN CA-resampler stability and full-grid audit

After the first positive CA-resampler result, the branch was stress-tested with a low-LR rerun and a full `3 x 5` timestep/layer grid.

Completed execution:

- original pending low-LR job `86444` was cancelled because its `128G` memory request blocked scheduling despite available GPU opportunities;
- replacement low-LR job `86453` ran on `air-node-02`, exit `0:0`, elapsed `01:10:59`, with `WAN_LR=5e-5`, `WAN_VAL_EVERY=250`, `t499/layer14`;
- full-grid job `86468` ran on `air-node-02`, exit `0:0`, elapsed `06:31:56`, with `WAN_PACK_PARALLEL=3`, `WAN_LR=1e-4`, and `timesteps=249,499,749 x layers=9,14,19,24,29`.

Key comparison:

| setting | F@0.10 | pred-to-GT p90 | Chamfer-L2 |
| --- | ---: | ---: | ---: |
| old hidden MLP `t499/layer09` | `0.46988987902779306` | `0.48718947172164917` | `0.21040735269586244` |
| CA historical peak `t499/layer14` from `86429` | `0.5012956284974035` | `0.4229188362757365` | `0.10908368105689685` |
| CA low-LR `t499/layer14` from `86453` | `0.4292316005720653` | `0.5338096991181374` | `0.15617707930505276` |
| CA full-grid `t249/layer14` from `86468` | `0.48913902331806663` | `0.4372795696059863` | `0.12182091859479745` |
| CA full-grid `t249/layer09` from `86468` | `0.4766100401399189` | `0.410525918006897` | `0.11981592203179996` |
| clean-GT VGGT layer16 | `0.6860468604251301` | `0.20770130679011345` | `0.026904070439438026` |

Full-grid result:

| timestep | layer09 | layer14 | layer19 | layer24 | layer29 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 249 | `0.47661004` | `0.48913902` | `0.45520595` | `0.44394081` | `0.43922030` |
| 499 | `0.42516608` | `0.42950700` | `0.42602896` | `0.41505938` | `0.37056562` |
| 749 | `0.41729879` | `0.45614139` | `0.43892215` | `0.41276225` | `0.38228251` |

Interpretation:

- CA-resampler remains the only clearly positive WAN readout family, but the original `t499/layer14=0.5013` should be treated as a historical peak rather than a stable mean.
- The low-LR rerun did not stabilize or improve the setting; it is a negative result.
- The full-grid repeat shifts the most reliable WAN settings toward `t249/layer09/14`. `t249/layer14` is best by F@0.10, while `t249/layer09` is best by p90 and Chamfer.
- Later layers `24/29` remain weak as single-layer probes.
- WAN still trails clean-GT VGGT by a large margin, so the next useful work is repeat/visual validation of `t249/layer09`, `t249/layer14`, and historical `t499/layer14`, not another broad single-layer sweep.

## 2026-05-17 WAN learned CA-resampler readout audit

The first generator-preserving learned readout audit completed after the negative pred-x0 and fixed-grid checks.

Implemented adapter:

- `WanHiddenCrossAttentionResamplerAdapter` / `--adapter_type wan_cross_attn_resampler`;
- valid only for WAN hidden cache (`--wan_feature_kind hidden`);
- restores hidden cache `[B,3120,1536]` as `[B,2,30,52,1536]`;
- adds learnable temporal / row / column position embeddings;
- cross-attends `768` learnable NOVA query tokens to WAN hidden tokens and outputs `[B,768,128]`.

Completed execution:

- smoke job `86427` completed successfully;
- formal layer09 job `86428` completed on `air-node-02`, exit `0:0`, elapsed `00:54:48`;
- layer sweep job `86429` completed on `air-node-02`, exit `0:0`, elapsed `02:14:39`, covering `t499 x layers 14,19,24,29`; layer09 is covered by `86428`;
- user inspected the layer14 `val_visual_40960/step_011500/` previews and reported visual improvement.

Layer sweep results:

| setting | F@0.10 | pred-to-GT p90 | Chamfer-L2 |
| --- | ---: | ---: | ---: |
| old hidden MLP `t499/layer09` | `0.46988987902779306` | `0.48718947172164917` | `0.21040735269586244` |
| CA resampler `t499/layer09` | `0.4675106769755634` | `0.4791356101632118` | `0.1388903713474671` |
| CA resampler `t499/layer14` | `0.5012956284974035` | `0.4229188362757365` | `0.10908368105689685` |
| CA resampler `t499/layer19` | `0.45041048278489915` | `0.5866823519269625` | `0.192564707249403` |
| CA resampler `t499/layer24` | `0.408836804475268` | `0.5937060564756393` | `0.212364894648393` |
| CA resampler `t499/layer29` | `0.3833416179630764` | `0.5307636285821596` | `0.25971310896178085` |

Interpretation at the time: the WAN path is live and the readout/interface matters. CA-resampler `t499/layer14` improved over old hidden MLP `t499/layer09` by `+0.03140575` F@0.10, `-0.06427064` pred-to-GT p90, and `-0.10132367` Chamfer-L2. The stability/full-grid audit on `2026-05-18` later showed this is a historical peak rather than a stable mean, but it remains the highest WAN metric seen so far.

Follow-up state: low-LR and full-grid reruns are recorded in the `2026-05-18` section above.

## 2026-05-16 WAN pred-x0 and hidden-grid readout audits

The WAN predicted-x0 / denoised-latent branch moved from cache generation into a readout-specific training audit.

Completed execution:

- full pred-x0 cache job `86411` completed with `329/329` samples under `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81_pred_x0_latent/t499/pred_x0_latent/`;
- cache tensors have expected shape `[12480,16]` with metadata including `sigma`, `x0_formula`, and latent/model-output shapes;
- original pred-x0 MLP readout failed before producing a formal result: `86412` hit a PyTorch CUDA `AdaptiveAveragePooling` backward assert when pooling `[12480,128] -> [768,128]`;
- `WanPredX0LatentConvAdapter` / `--adapter_type conv2d_mlp` was added to restore the latent grid `[B,2,60,104,16]`, apply a stride-2 Conv2d readout to `[B,3120,128]`, then pool to `[B,768,128]`;
- smoke job `86421` completed on `air-node-02`, exit `0:0`, with robust validation and no pooling crash;
- formal job `86422` completed on `air-node-02`, exit `0:0`, elapsed `00:52:13`, output `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_wan_t2v1p3b_ctx81_pred_x0_latent_conv2d_t499`.

Pred-x0 Conv2d result:

- `best_val_fscore_tau_0.10=0.41336424426176693`
- `best_val_pred_to_gt_p90=0.6272750149170557`
- `best_val_chamfer_l2=0.35522504647572833`

Hidden-grid readout audits:

- `WanHiddenGrid2DPoolAdapter` / `--adapter_type grid2d_pool` preserves the hidden cache as `[B,2,30,52,1536]`, maps channels token-wise, then pools to NOVA `[B,2,24,16,128]`; smoke/formal jobs `86423` / `86424` completed on `air-node-04`, with `F@0.10=0.43326753863230877`, `pred_to_gt_p90=0.5651116619507471`, and `Chamfer-L2=0.1759416777640581`;
- `WanHiddenGrid2DConvAdapter` / `--adapter_type grid2d_conv` adds a small same-resolution 3x3 Conv2d readout before the same 2D pooling; smoke/formal jobs `86425` / `86426` completed on `air-node-04`, with `F@0.10=0.39855373112050535`, `pred_to_gt_p90=0.7689011543989182`, and `Chamfer-L2=1.159957120815913`.

Interpretation at the time: old WAN Route2 `t499/layer09` remained the best WAN setting (`F@0.10=0.46988987902779306`, `pred_to_gt_p90=0.48718947172164917`). Pred-x0 latent, fixed 2D hidden pooling, and tiny fixed 2D conv readout were all negative under the current NOVA/FM generator setup. This motivated the learned CA-resampler branch recorded in the `2026-05-17` section above, which later surpassed the old WAN hidden baseline.

## 2026-05-15 WAN Route2.1 result, normalization / context checks, and pred-x0 queue

The WAN Route2.1 clean / low-noise audit finished after the original 2026-05-12 queue.

Completed execution:

- cache smokes: `86342` (`low_noise`) and `86343` (`no_noise`);
- full caches: `86350` and `86351`;
- smoke train: `86357`;
- formal packs: replacement 1-GPU jobs `86371` (`no_noise`) and `86372` (`low_noise`), both completed successfully on `air-node-04`.

Result summary:

- best Route2.1 clean/low-noise result was `no_noise/layer29` with `best_val_fscore_tau_0.10=0.44840173` and `best_val_pred_to_gt_p90=0.48779308`;
- this did not beat the old Route2 best `t499/layer09` (`F@0.10=0.46988987902779306`, `pred_to_gt_p90=0.48718947172164917`);
- therefore the WAN weakness is unlikely to be explained only by noisy latent diffusion states.

Completed follow-up checks:

- `train_wan_t2v_nova_adapter.py` now supports `--wan_feature_norm none|token_layernorm|sample_standardize|token_l2`;
- `slurm/scrream_wan_t2v_ablation_pack_train.sbatch` exposes this as `WAN_FEATURE_NORM`;
- targeted job `86395` completed `t499/layer09 + token_layernorm` with `best_val_fscore_tau_0.10=0.45476213575933216`, `best_val_pred_to_gt_p90=0.484821617603302`, and `best_val_chamfer_l2=0.1771892917652925`;
- `pair_tiled81` vs `ctx81` completed as jobs `86396 -> 86397 -> 86400 -> 86401` with `best_val_fscore_tau_0.10=0.4429200036613287`, `best_val_pred_to_gt_p90=0.5104739194115003`, and `best_val_chamfer_l2=0.16478415516515574`;
- neither feature normalization nor pair-tiled context beat the old hidden-state Route2 best `t499/layer09`.

Predicted-x0 / denoised-latent follow-up:

- `prepare_scrream_wan_t2v_feature_cache.py` now supports `--feature_kind pred_x0_latent`, computing `x0 = sample - sigma * model_output` from WAN's `flow_prediction` scheduler path;
- pred-x0 caches use `t499/pred_x0_latent/<sample>.pt`, expected feature shape `[12480,16]`, and metadata records `sigma`, scheduler class / prediction type, source latent shape, model output shape, and `x0_formula`;
- `train_wan_t2v_nova_adapter.py` supports `--wan_feature_kind pred_x0_latent`; the original MLP-L4 readout was later superseded by the 2026-05-16 Conv2d readout above after the MLP pooling path failed.

Adapter-capacity note: the current MLP adapter already contains `GELU`, so future `MLP-L6-H1024`, `MLP-L4-H2048`, or cross-attention checks should be treated as low-priority capacity/readout ablations rather than the main suspected cause.

## 2026-05-12 WAN Route2 completion and setting audit

The WAN Route2 branch was sanity-checked and then completed as a full 15-run ablation pack.

Key results from job `86316`:

- the WAN caches are not degenerate: real WAN features differ materially from the zero-feature and sample-shuffle controls;
- mean feature relative L2 to the reference config is `1.4833955196788304`;
- mean adapter-token relative L2 to the reference config is `0.9976405603462312`;
- zero-control `t499/layer09` reached `best_val_fscore_tau_0.10=0.38500468702435137`, `best_val_chamfer_l2=0.8643565624952316`, and `best_val_pred_to_gt_p90=0.47307824591795605`;
- sample-shuffle `t499/layer09` reached `best_val_fscore_tau_0.10=0.37489499366133267`, `best_val_chamfer_l2=0.682008907198906`, and `best_val_pred_to_gt_p90=0.6679268131653467`.

Formal Route2 training pack:

- job `86307` / `scrream_wan_t2v_pack`: `COMPLETED`, exit `0:0`, elapsed `13:36:04`, node `air-node-04`;
- full grid completed: timesteps `249,499,749` x code layers `9,14,19,24,29`;
- best WAN run: `t499/layer09`, `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, `best_val_chamfer_l2=0.21040735269586244`;
- clean-GT VGGT layer `16` remained much stronger with `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, and `best_val_chamfer_l2=0.026904070439438026`.

Interpretation:

- the WAN route is live, but the absolute scores are still much worse than clean-GT VGGT;
- the current evidence is strong enough to rule out a trivial cache-loader bug, but not strong enough to support a claim that WAN is a good geometric backbone in this probe setting;
- the immediate follow-up moved into execution on `2026-05-12`: Route2.1 `no_noise` / `low_noise` support was implemented, and jobs `86342/86343 -> 86350/86351 -> 86357 -> 86358/86359` were queued for layers `9,14,29`; this queue was later superseded by the completed 2026-05-15 state above.

## 2026-05-11 SCRREAM sequence-meta GT correction and clean VGGT rerun

The mesh-complete SCRREAM GT was found to include scene-level objects that were not present in some reduced sequences. The concrete audit case was `scene08/scene08_reduced_00_000220_000260`, where mannequin OBJ files existed under `scene08/meshes/` but were absent from `scene08_reduced_00/meta.txt`.

Code/data correction:

- `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py` now reads each sequence `meta.txt`.
- `mesh_complete` samples only OBJ files whose stem is listed in the current sequence metadata.
- mesh cache labels now include `scene + sequence + object-set hash`, preventing reuse of old scene-level reservoirs.
- old pre-meta-filter `.pt` files were moved to `experiments/probe3d/adapter_data/deprecated_meta_filter_bug/`.

Regeneration jobs:

- `86283` / `scrream_mesh_prep`: `COMPLETED`, exit `0:0`, elapsed `00:51:13`
- `86284` / `scrream_split_merge`: `COMPLETED`, exit `0:0`, elapsed `00:00:29`
- output: `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
- shape / split: `[329, 20000, 3]`, `train=317`, `val=12`
- metadata: `mesh_sequence_meta_filter=True`

Known sample check:

- `scene08/scene08_reduced_00_000220_000260`
- old selected mesh count: `25`, including three mannequin OBJ files
- new selected mesh count: `22`
- excluded by sequence metadata: `human-female_mannequin_green_top.obj`, `human-female_mannequin_grey_top.obj`, `human-male_mannequin_colorful_top.obj`

Clean-GT VGGT rerun:

- `86285` requested `4 x A100` but was cancelled because Slurm predicted a long resource wait.
- `86286` completed on `air-node-04` with exit `0:0`, elapsed `02:23:11`.
- output prefix: `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_vggt_layerXX`
- job `86286` uses the 30-epoch-equivalent `9510` steps because it was submitted before the default epoch change.

Clean-GT layer result:

| layer | best F@0.10 ↑ | best pred→GT p90 ↓ | best Chamfer-L2 ↓ | interpretation |
| ---: | ---: | ---: | ---: | --- |
| 16 | **0.6860468604251301** | **0.20770130679011345** | **0.026904070439438026** | current default |
| 24 | 0.6802513448244976 | 0.25152526050806046 | 0.03554012098660072 | main comparison point |
| 20 | 0.548158719135383 | 0.36711231619119644 | 0.0767408860847354 | no longer default under clean GT |

The sequence-meta correction changed the layer ranking: pre-meta-filter job `86149` favored layer `20`, but the clean target favors layer `16`, with layer `24` close on F-score and better final GT-to-pred recall.

Training-length policy changed after this submission: formal SCRREAM VGGT/WAN ablation defaults now use `50` epochs unless explicitly overridden.

## 2026-05-10 WAN Route2 setup

The completed SCRREAM VGGT layer ablation originally motivated a WAN2.1 T2V representation probe. After the 2026-05-11 sequence-meta GT correction, the old VGGT result is pre-meta-filter history; WAN should be compared against the clean-GT VGGT rerun once it finishes. The WAN branch is designed to keep the training problem fixed and swap only the representation.

Fixed setting inherited from the VGGT ablation:

- adapter data: `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
- split: `train=317`, `val=12`
- GT: sequence-meta-filtered registered SCRREAM mesh-complete surfaces, two-view union-frustum crop, first input camera frame, `20000` points
- adapter/loss/decoder: `MLP-L4-H1024`, `nova_flow`, NOVA `scene_ae`
- validation: full val split, robust one-way/F-score/trimmed-CD metrics, `val_visual_40960/`

WAN Route2 implementation added:

- `third_party/VidFM3D` as a reference submodule
- `experiments/probe3d/requirements-wan-t2v.txt`
- `experiments/probe3d/scripts/prepare_scrream_wan_t2v_feature_cache.py`
- `experiments/probe3d/train_wan_t2v_nova_adapter.py`
- `slurm/scrream_wan_t2v_download.sbatch`
- `slurm/scrream_wan_t2v_precompute.sbatch`
- `slurm/scrream_wan_t2v_ablation_pack_train.sbatch`

WAN representation grid:

- model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- prompt: empty string
- context: 81 RGB frames around each SCRREAM pair
- timesteps: `249,499,749`
- code layer ids: `9,14,19,24,29`
- cached tensor shape: `[3120,1536]`

Preflight and execution status:

- the isolated WAN dependency set imported successfully in `nova3r` with `diffusers 0.37.1`, `transformers 4.57.6`, and `tokenizers 0.22.2`;
- window-only validation passed for all `329` SCRREAM samples;
- WAN repo/checkpoint jobs use proxy `http://127.0.0.1:17890` through the compute-node SSH tunnel to `air-server:127.0.0.1:17890`;
- checkpoint download/preflight completed; the local checkpoint is `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers` (~27 GB);
- 2-sample feature smoke job `86282` completed for timestep `749`, layer `20`, writing two `[3120,1536]` FP16 cache files;
- full feature precompute jobs `86292`, `86293`, and `86294` completed for timesteps `249`, `499`, and `749`;
- feature cache root `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81` contains `4937` `.pt` files / about `45G`;
- official grid coverage is complete at `329` samples for every `(timestep, layer)` in `249,499,749 x 9,14,19,24,29`;
- first training pack attempt `86306` failed immediately on a reduced-loss float `.item()` bug;
- `experiments/probe3d/train_wan_t2v_nova_adapter.py` was fixed by keeping `final_loss = reduced_loss`;
- replacement pack job `86307` completed on `air-node-04`, one A100, 50 epochs / `15850` steps per run, exit `0:0`, elapsed `13:36:04`.

## 2026-05-07 SCRREAM 20k / 500k adapter run

The SCRREAM mesh-complete line advanced from data-prep handoff to a completed 20k / 500k MLP baseline.

Generated adapter datasets:

- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt`
  - shape `[329, 10000, 3]`
  - split `train=223`, `val=12`, `test=94`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_trainplus_test.pt`
  - split `train=317`, `val=12`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000.pt`
  - shape `[329, 20000, 3]`
  - split `train=223`, `val=12`, `test=94`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
  - split `train=317`, `val=12`

The `trainplus_test` datasets are split-label rewrites: `test` is converted to `train`, and `val` is preserved. They should not be used for held-out test claims.

The completed training job is:

- `86140` / `scrream_mesh_mlp`
- state: `COMPLETED`, exit `0:0`
- node: `air-node-02`
- GPU: `1 x A100`
- elapsed: `00:26:26`
- Slurm end time: `2026-05-07 22:40:22 CST`
- input `.pt`: `scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
- output dir: `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17`
- SwanLab run: `https://swanlab.cn/@JiachengDong/PSUVPSC3DD/runs/eoiupi3bv2g11dvd21ypm`
- config: `MLP-L4`, hidden `1024`, `nova_flow`, `num_queries=20000`, requested `9510` steps

Final metrics:

- `first_loss=1.4599288702011108`
- `final_loss=0.8398033976554871`
- `best_loss=0.5998285412788391`
- `best_val_chamfer_l2=0.5149603486061096`

The latest `validation_metrics.json` is step `9500` with `val_chamfer_l2=0.6779176592826843`; the best validation value is the one recorded in `final_metrics.json`.

Operational fixes during this run:

- `third_party/vggt` had to be initialized after conversion to a submodule; an empty submodule directory caused `ModuleNotFoundError: No module named 'vggt.models.vggt'`.
- `slurm/scrream_merge_test_into_train.sbatch` was added to convert `test -> train` safely and run input checks.
- `slurm/scrream_mesh_complete_mlp_train.sbatch` now supports single-node DDP via `SCRREAM_GPUS_PER_NODE`; future multi-GPU runs should set `SCRREAM_EPOCHS` so step count scales by effective batch size.
- Local `scrream_official_depth_mix_*` artifacts are treated as historical ablations and are not part of the current training line.

## 2026-05-03 SCRREAM full-data mesh-complete bridge

The corrected SCRREAM branch is active again because the full dataset is now present at:

- `~/datasets/SCRREAM`

This does not revive the old `eval_scrream` results. Those remain invalid for formal claims because they used the released evaluation subset. The corrected branch uses the full SCRREAM directory structure and the official two-view pair list:

- `data/scrream/scrream_n2_list.json`

### Implemented data bridge

Added and extended:

- `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`

The script writes adapter `.pt` datasets compatible with `AdapterImagePointDataset`:

- `scene_ids`
- `target_points`
- `splits`
- per-sample `metadata`
- global `meta`

The training loader was also corrected so:

```bash
python experiments/probe3d/train_vggt_nova_adapter.py \
  --dataset scrream_adapter \
  --data_root experiments/probe3d/adapter_data/<adapter_dataset>.pt
```

actually reads the requested `.pt` instead of falling back to the default adapter path.

### Target-source decision

The first bridge supported `depth_gt_dense`: aggregate `depth_gt` frames between the two input frames, voxel-deduplicate, crop to the input frustum, and sample a fixed target.

After discussion, this is now an alternate baseline rather than the main path. Dense depth aggregation can include surfaces not visible in the first input frame when intermediate frames saw them, but it is still limited by observed depth frames. It is not a truly complete mesh target.

The active target source is now:

- `--target_source mesh_complete`

Mesh-complete semantics:

1. read `sceneXX/meshes/*.obj`
2. sample scene mesh surfaces
3. voxel-deduplicate and cache a per-scene reservoir
4. crop world points to the selected input pair's union frustum
5. transform the final target into the first input camera frame
6. deterministic FPS to `10000` points

This is closer to the desired adapter target: complete / amodal geometry inside the selected input-view frustum, without asking the model to reconstruct the entire room outside that frustum.

### Area-proportional mesh sampling correction

The first mesh-complete preview sampled roughly equal point counts per OBJ. That made large background assets such as walls and room shells too sparse compared with small object meshes.

The mesh reservoir was corrected to allocate samples proportional to estimated mesh surface area. Example effect for `scene09`: the room/background mesh receives the dominant share of samples, while smaller objects receive smaller budgets. The accepted preview path is:

- `experiments/probe3d/adapter_data/scrream_mesh_complete_area_n2_preview/scene09_mesh_complete_area_world_reservoir.ply`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_area_n2_preview/scene09_mesh_complete_area_first_view_reservoir.ply`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_area_n2_preview/scene09__scene09_full_00_000200_000275_target_first_view.ply`

The user visually accepted this preview, so area-proportional mesh sampling is the current default.

### Slurm handoff

The machine uses Slurm. New job scripts:

- `slurm/scrream_mesh_complete_prepare.sbatch`
- `slurm/scrream_mesh_complete_mlp_train.sbatch`
- `slurm/scrream_mesh_complete_mlp_smoke.sbatch`

Logs are written to:

- `slurm_out/`

The Slurm scripts default proxy variables to `http://127.0.0.1:7896`, matching the working network path used for checkpoint downloads and SwanLab access.

Checkpoint and tracking setup completed:

- `checkpoints/scene_n1/checkpoint-last.pth` and `.hydra/config.yaml`
- `checkpoints/scene_n2/checkpoint-last.pth` and `.hydra/config.yaml`
- `checkpoints/scene_ae/checkpoint-last.pth` and `.hydra/config.yaml`
- `checkpoints/vggt/model.pt`
- `swanlab==0.7.16` imports in the `nova3r` conda env

The SCRREAM training path now normalizes adapter `.pt` targets with the NOVA decoder checkpoint's `norm_mode` before the `nova_flow` loss. The local `scene_ae` metadata reports `norm_mode=median_3`.

Full data generation and dependent training have now been submitted:

- `85773` / `scrream_mesh_prep`: running on `air-node-04` at 2026-05-03 02:26 CST
- `85774` / `scrream_mesh_mlp`: pending on dependency after `85773`

At that timestamp, the full output files had not yet been written:

- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.manifest.json`

Status commands:

```bash
squeue -j 85773,85774 -o '%.18i %.30j %.8T %.10M %.9l %.30R'
sacct -j 85773,85774 --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End,NodeList%20
```

## 2026-04-29 correction block — interval and metric

A major audit corrected the interpretation of the ScanNet runs:

- The processed ScanNet data already uses `frame_skip=20`. The inherited dataloader `max_interval=30` therefore sampled views up to roughly 600 raw frames apart. All old K-view conclusions from those runs are interval-confounded.
- Corrected local-overlap experiments use `scannet_max_interval=1`, equivalent to adjacent processed frames / roughly 20 raw frames.
- GT-only visual audit showed the three ScanNet target modes (`anchor_frustum`, `covered_by_ge2`, `nova_input_frustum`) are not drastically different by eye.
- Oracle CD rankings were dominated by tiny sample counts, stochastic decoder sampling, flow-vs-CD mismatch, and outlier samples; they are no longer treated as reliable target-mode rankings.
- Robust evaluation of the interval-corrected MLP baseline shows moderate recall but poor precision/outlier control, matching the qualitative failure.


This document records what actually happened, including corrections.

## 2026-04-29 evening handoff update

The local cleanup branch is now `wip/psuvpsc3dd-probe-20260429` for server handoff. Push that branch before using the handoff commands in `docs/probe/handoff_2026-04-29.md`.

Fixed-30 robust eval of the K2/interval=1 MLP-L4/chamfer baseline confirmed the main failure mode: F@0.05 mean/median `0.291/0.275`, precision@0.05 mean `0.204`, recall@0.05 mean `0.532`, with strong prediction-side outlier issues.

A lightweight K2/interval=1 cross-attention candidate (`p7_k2_i1_anchor_ca_l2_h512_chamfer_step1000`) completed 1000 steps with validation CD `0.54222615`. This is not a scalar improvement over MLP and should only be interpreted after fixed-30 robust eval/renders.

After this diagnosis, the 2026-04-29 planned data direction was an InteriorGS-style high-quality indoor 3DGS pilot on this server. That plan is now deferred because the full SCRREAM dataset became available locally on 2026-05-03 and the mesh-complete SCRREAM branch became the active training line.

## 1. Major correction — old SCRREAM results are invalid for claims

A later audit found that the local `eval_scrream` package used in the earlier SCRREAM line was only the released evaluation subset, not the official full-data setup.

Therefore:

- the older SCRREAM quantitative results are **invalid as formal evidence**
- they remain useful as **engineering/debugging history** only
- they must not be used as proposal feasibility claims

## 2. What remains valid from the earlier history

Even after that correction, the repo still accumulated real engineering progress:

- adapter branches were implemented and trained
- NOVA3R-style decoder integration was established
- caching / logging / export paths were exercised
- debugging knowledge compounded across multiple runs

So the history is not wasted — it just must be interpreted honestly.

## 3. New active branch — ScanNet v2 mesh-first extension

After the SCRREAM correction, a new formal branch became the active scale-up path.

### Interpretation
This branch is:
- a **NOVA3R-style extension / transfer probe** on ScanNet v2
- not a literal reproduction of official NOVA3R scene training on `3D-FRONT + ScanNet++V2`

### Formal data decisions
- mesh source: `vh_clean.ply`
- frame sampling: `frame_skip=20`
- GT: `mesh surface ∩ sparse-input-view union frustum`
- no extra visible/occluded auxiliary labels
- reservoir density: `500k / scene`

### Formal processed dataset
- processed root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k`
- formal split root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17`
- scene counts:
  - `train=1362`
  - `val=151`
  - `test=100`

### Key implementation milestones
- true 4-view input wiring for ScanNet batches
- `pts3d_complete` dataset path from scene-level mesh reservoirs
- complete-GT builder script added
- DDP / `torchrun` conversion for MLP / CA / SA

## 4. Smoke / preflight milestones

### Complete-GT ScanNet smoke
- output dir:
  - `experiments/probe3d/result/scannet_mlp_complete_smoke_seed17`
- SwanLab run id:
  - `30iyh29o1orq7mnm097u4`

### Full-root MLP DDP preflight
- output dir:
  - `experiments/probe3d/result/scannet_mlp_ddp_fullroot_preflight_seed17`
- SwanLab run id:
  - `f7nivqbjlilypp5o9u4sz`

These runs matter because they show the current formal branch is not just a paper plan:

- full preprocess exists
- full split exists
- complete GT exists
- DDP launch path exists
- the full-root training path actually starts and runs

## 5. Superseded formal baseline run

The earlier formal run target was the **ScanNet MLP baseline**. This remains part of the history, but the active direction was later superseded by the shorter ScanNet probe in Section 6.

### Purpose
This is not meant as the final headline method.
Its purpose is to establish whether:

- frozen VGGT features
- a lightweight MLP adapter
- and a NOVA3R-style decoder

can learn stable complete 3D behavior under reliable mesh-first supervision at scale.

### Formal schedule
- 8 GPUs
- batch size `1 / rank`
- 50 epochs
- `steps_per_epoch = 13158`
- `max_steps = 657900`
- validation/checkpoint every 5 epochs

### Formal output dir
- `experiments/probe3d/result/scannet_mlp_adapter_l4_lr5e-5_seed17_epoch50_formal`

## 6. Short ScanNet probe update — objective mismatch isolated

A short harness was added under:

- `experiments/probe3d/probe_trials/`

The purpose was to stop relying on a long formal run and instead test target modes / objectives quickly with immutable logging in `results.tsv`.

### Target-mode findings

Oracle token optimization showed that target support matters:

- `anchor_frustum`: oracle CD `0.23668508`
- `anchor_frustum_margin1.5`: oracle CD `0.81146899` — too broad / out of generator support
- `covered_by_ge2`: oracle CD `0.18723274`
- `covered_by_ge2_anchorfb`: oracle CD `0.20196436`

`anchor_frustum` became the stable adapter target because it avoided empty-support / fallback contamination and stayed within the anchor-camera generator domain.

### Objective findings

With `anchor_frustum`, the old `nova_flow` objective underperformed:

- `MLP-L4 + nova_flow`, step1000: `0.34512938`
- `MLP-L2-h512 + nova_flow`, step1000: `0.30276422`

Switching to direct sampled rollout Chamfer was the decisive change:

- `MLP-L4 + chamfer_sample`, step1000: `0.11552816`
- continued to step2000 at `lr=5e-5`: `0.09181590`
- continuing at `lr=5e-5` became unstable / overfit by step2500: `0.35143724`, so it was stopped
- resuming from step2000 with `lr=1e-5` refined to step2500: `0.08745259` — current best
- continuing the same low-LR branch to step3000 slightly worsened: `0.09132496`

### Visual / diagnostic finding

The numeric best is not yet visually satisfying. A GT-vs-pred video showed that the model covers much of the target support but produces a thick / loose / outlier-heavy point cloud.

Nearest-neighbor diagnostic on the visualized sample:

- `GT→Pred` mean distance: `0.0582`
- `Pred→GT` mean distance: `0.1777`

This means recall is acceptable but precision is poor. The current bottleneck is no longer simply “make CD go down”; it is prediction sharpness / outlier suppression.

### Resulting conclusion

The key learning is: direct rollout Chamfer fixes a large train/eval objective mismatch, but symmetric Chamfer alone can be gamed by noisy coverage. The next method change should be precision-aware: overweight `pred→GT`, use trimmed Chamfer, or add an outlier penalty.

## 7. Bottom line

The cleanest honest reading of the repo on 2026-04-29 is:

- old SCRREAM eval-subset claims are invalid
- the engineering stack survived that correction
- the ScanNet v2 mesh-first line was the main trustworthy scale-up path before the full SCRREAM data became available locally on 2026-05-03
- the current best numeric baseline is `anchor_frustum + MLP-L4 + direct Chamfer`, CD `0.08745259`
- that numeric result is not visually clean enough: recall is acceptable, precision / outlier control is poor
- at that timestamp, the next milestone was an InteriorGS data bridge and visual sanity pass before new training; this was later superseded on 2026-05-03 by the full SCRREAM mesh-complete branch

## 8. Paper-aligned GT / loss correction after user review

Jiacheng clarified that NOVA3R's completion target is not a full-room point cloud. It is a complete / amodal point cloud **within the selected input-view frustum**: for example, if a table is in view, the model should recover surfaces such as the underside of the table within that frustum, not hallucinate the entire room.

This changes the interpretation of the previous `anchor_frustum` result. It was useful because it partially aligned with NOVA3R's target construction, but for K-view training the paper-aligned target should be the union of the selected input frusta, not just the first view. Direct Chamfer remains a diagnostic, while the main representation-probe objective should return to NOVA-native flow matching.

Implementation started:

- added explicit `nova_input_frustum` alias for union-of-selected-input-frusta complete targets
- added `nova_anchor_frustum` alias for K=1 / first-view debug targets
- exposed `scannet_complete_points` through loader, oracle, adapter, and the probe harness
- added `query_source` passthrough in the harness so trials can use `src_complete_fps_4096`
- created `experiments/probe3d/probe_trials/configs/phase2_nova_aligned.json` with K=1/K=2 oracle and MLP-L4 native-flow adapter trials

Next result to record: K=1/K=2 oracle support for `nova_input_frustum + src_complete_fps_4096`.


## 9. Phase-2 paper-aligned frustum oracle controls

After Jiacheng's clarification that NOVA3R predicts complete/amodal geometry constrained to input-view frusta rather than whole-room completion, I added explicit target modes and controls for more paper-literal target construction.

Implementation notes:

- `nova_input_frustum`: union of the selected input-view frusta, collapsed into the existing `pts3d_complete` target path.
- `nova_per_view_frustum`: each input view contributes its own complete frustum crop, matching NOVA's `get_complete_pts3d()` per-view stacking convention more literally.
- `nova_per_view_frustum_anchor_zpos`: same as per-view but clipped to positive anchor-camera z for ScanNet stability.
- Exposed `scannet_complete_points` and `query_source` through the probe harness.
- Fixed two runtime issues discovered by controls: ScanNet `num_views=1` sequence sampling crashed in `get_seq_from_start_id`, and local PyTorch3D FPS requires an explicit `max_K` argument.

Oracle results on the first two val samples were not promising:

- `nova_input_frustum`, K1, `src_complete_fps_4096`: mean CD `0.4766`
- `nova_input_frustum`, K2, `src_complete_fps_4096`: mean CD `1.3288`
- `nova_input_frustum`, K4, `src_complete`: mean CD `0.7692`
- `nova_input_frustum`, K4, `src_complete_fps_4096`: mean CD `1.3315`
- `nova_per_view_frustum`, K4, `src_complete_fps_4096`: mean CD `1.6023`
- `nova_per_view_frustum_anchor_zpos`, K4, `src_complete_fps_4096`: mean CD `0.9611`

Key interpretation: K1 controls are not directly comparable to the real 4-view probe because the NOVA decoder receives `num_views=1` conditioning, which appears out-of-domain. For the actual 4-view probe, the paper-literal union/per-view input-frustum targets fail oracle gating. The earlier 4-view `anchor_frustum` / `covered_by_ge2` targets remain more credible generator-domain targets (`anchor_frustum` oracle around `0.2367`, `covered_by_ge2` around `0.1872`). Therefore the next adapter experiments should not use `nova_input_frustum` yet; they should use oracle-supported 4-view target definitions and focus on native-flow or hybrid objective design.

## 10. Phase-4 feasibility pivot — K=2, MLP-L4, loss and GT sweep

After additional user review, the current priority was changed from exact NOVA3R GT reproduction to a cleaner **feasibility proof**: show that frozen VGGT representations can be adapted into a latent/token form that the NOVA generator can consume and roll out into meaningful 3D.

### Why the pivot happened

Two observations motivated the new sweep:

- The exact official NOVA3R GT construction is not fully observable from the released code/data path, so exact alignment should not block feasibility validation.
- Jiacheng pointed out that increasing the number of input views can itself reduce / empty the valid complete-target support under some target packagings. Quick loader stats supported this concern: K=4 often introduced empty view slots or lower average valid target counts than K=2.

### Completed quick controls before the overnight run

- `p4_oracle_norm_nova_per_view_ldi4_k2_fps4096_s2_step600`: CD `0.78956747`
- `p4_oracle_norm_anchor_frustum_k2_fps2048_s2_step600`: CD `0.10993152`

Interpretation:

- K=2 did **not** rescue the current LDI-style per-view target.
- K=2 made the stable `anchor_frustum` feasibility target much more generator-reachable than the previous K=4 setting.

### Active overnight loss ablation

The current active run is the K=2 / MLP-L4-H1024 / `anchor_frustum` loss ablation:

- `p4_k2_anchor_mlp_l4_flow_step1000` — native `nova_flow`
- `p4_k2_anchor_mlp_l4_hybrid005_step1000` — `nova_flow + 0.05 * rollout_chamfer`
- `p4_k2_anchor_mlp_l4_chamfer_step1000` — direct rollout Chamfer diagnostic

Driver log:

- `experiments/probe3d/result/probe_trials/p4_k2_mlp_l4_loss_ablation_driver.out`

As of the documentation update, the first run was active around step ~676.

### Queued overnight GT-construction sweep

A continuation script waits for the active MLP-L4 loss ablation to finish and then launches additional K=2 controls.

Script / driver:

- script: `experiments/probe3d/result/probe_trials/p4_overnight_after_l4_ablation.sh`
- driver log: `experiments/probe3d/result/probe_trials/p4_overnight_after_l4_ablation_driver.out`
- launcher PID at creation time: `339478`

Queued oracle GT candidates:

- `nova_input_frustum`
- `covered_by_ge2`
- `anchor_frustum_margin`, margin `1.5`
- `nova_per_view_frustum_anchor_zpos`
- `nova_per_view_ldi2`
- `nova_per_view_ldi4`
- `nova_per_view_ldi8`

Queued adapter target/loss candidates:

- targets: `covered_by_ge2`, `nova_input_frustum`, `anchor_frustum_margin1.5`
- losses: native `nova_flow` and `flow_chamfer_hybrid` with `chamfer_weight=0.05`
- adapter: `MLP-L4-H1024`
- views: K=2
- queries: `2048`

### How to interpret the 2026-04-29 follow-up results

The primary feasibility claim should come from the best K=2 result that satisfies both conditions:

1. The target has a plausible oracle ceiling.
2. The VGGT adapter approaches that ceiling enough to show the representation can be consumed by the NOVA generator.

Direct Chamfer remains a metric/visual diagnostic. Native flow and the small hybrid loss are more relevant to a generator-native representation claim.

### Step-count adjustment

Jiacheng requested that the training runs use slightly more steps. The short K=2 MLP-L4 native-flow run had already completed at step1000 with CD `0.91071419`, which is weak and should be treated as a partial / short-run diagnostic.

The overnight plan was updated to longer adapter schedules:

- resume `p4_k2_anchor_mlp_l4_flow_step1000` to `p4_k2_anchor_mlp_l4_flow_resume_step2000`
- run `p4_k2_anchor_mlp_l4_hybrid005_step2000`
- run `p4_k2_anchor_mlp_l4_chamfer_step2000`
- run the queued adapter GT/loss sweep at `2000` steps per trial instead of `1000`

New long-suite driver:

- `experiments/probe3d/result/probe_trials/p4_overnight_k2_mlp_l4_long_driver.out`
