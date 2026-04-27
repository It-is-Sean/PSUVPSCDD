# Proposal digest

## Title

**Probing Image and Video Foundation Representations via Shared Complete-3D Decoding**

## Core question

Can a frozen image / geometry / video representation be decoded by a **shared complete-3D decoder** into a stable, scene-level complete 3D structure?

The proposal distinguishes two different notions of 3D content inside a representation:

1. **Directly readable local geometry** — information that a shallow probe can recover as point/depth/pose outputs.
2. **Scene-level complete-3D structure** — information that only becomes visible when a shared decoder expands a compact scene representation into a complete 3D scene.

## Proposed pipeline

### Stage 1 — Shared canonical decoder

Train or reuse a canonical point-latent autoencoder on complete point clouds:

- `E_canon(P*) -> z*`
- `D_canon(z*) -> P_hat`

For this branch, the canonical decoder is intentionally anchored to **NOVA3R Stage 1**.

### Stage 2 — Lightweight scene-token adapter

For each frozen backbone representation `r_M^l`:

- normalize tokens to a shared dimension
- use fixed learnable scene queries
- aggregate via shallow cross-attention
- project to the canonical latent token shape
- decode with the shared frozen decoder

### Stage 3 — Shared decoding vs direct baseline

Run the same frozen representations through:

- **shared complete-3D decoding probe**
- **direct 3D readout baseline**

Record both signals instead of collapsing everything to one scalar.

## Main variables to compare

- representation family: image / geometry / video
- feature layer: middle vs final, single-layer vs multi-layer
- video timestep: early/mid windows vs later steps
- low-view regime: `K in {1, 2, 4, 8}`
- unseen-region stress: visible vs unobserved completion quality
- probe capacity / decoder query budget / latent dimension

## Candidate backbone families

### Image / multi-view geometry side

- NOVA3R
- VGGT
- Depth Anything 3 (DA3)
- DUSt3R
- MASt3R
- Pi3

### Video side

- WAN2.1
- Open-Sora 2.0
- CogVideoX
- V-JEPA

## Main metrics

### Direct baseline

- dense point map error
- dense depth error
- relative pose error

### Shared decoding branch

- Chamfer Distance
- Earth Mover's Distance (if budget allows)
- Completeness / Coverage
- Normal Consistency
- visible-region reconstruction
- unseen-region completion

## Risks explicitly acknowledged in the proposal

- conclusions are **decoder-conditional**
- a strong adapter can hide representation differences
- complete 3D supervision is cleaner on some datasets than others
- cross-family comparison can still be imperfect even with a unified protocol

## What this branch should optimize for

- fast iteration on the probe interface
- clean separation between reusable NOVA3R parts and new probing logic
- easy experiment bookkeeping
- explicit fairness / stress-test configuration
- future extension to more decoder families without rewriting the whole tree
