# Proposed: MiniMax H3 Ref2VA (reference image + reference audio)

## Metadata

- Created: 2026-10-12
- Status: Proposed
- Completed: N/A

## ADR status

- Governing ADRs: None
- ADR impact: None expected — extends the existing MiniMax-H3 route; no new runtime or abstraction.

## Pre-implementation findings

**STATUS: READY** (one download, the Ref2VA transformer, is the only missing asset; no download done).

Scope: preparation only. Nothing implemented, no weights downloaded or converted, installed
`mlx-gen` untouched. Two small *metadata* JSONs (`transformer_ref/config.json`, its
safetensors index, 64 KB) were fetched to a scratch dir to compare tensor names.

## 1. Environment

| Item | Value |
|---|---|
| mlx-gen (installed) | 0.38.0, `uv tool` install from PyPI (no VCS: `direct_url.json` absent) at `~/.local/share/uv/tools/mlx-gen`; entry `~/.local/bin/mlxgen`; H3 code in `site-packages/mflux/models/minimax_h3/` |
| Matching source | `lpalbou/mlx-gen` `main` = tag `v0.38.0` = `99fb94d` |
| Python / MLX (installed) | 3.13.13 / mlx 0.32.3 |
| Python / MLX (dev worktree, from `uv.lock`) | 3.13.13 / mlx **0.31.0** (differs from installed) |
| uv / FFmpeg | 0.11.19 / 9.0.1 |
| Memory | 128 GiB unified; 92% free at audit |
| Disk | 274 GiB free on `/System/Volumes/Data` (93% used of 3.6 TiB) |
| HF cache | `HF_HUB_CACHE=~/models/mlx`; content-addressed shared blob store `~/models/mlx/blobs` (195 GiB total, sha256-named) |
| H3 snapshot | `~/models/mlx/models--MiniMaxAI--MiniMax-H3/snapshots/42ed227e…` — `transformer` 62 GiB (14 bf16 shards, 638 tensors), `text_encoder` 62 GiB (Qwen3-VL, 14 shards), `vae` 9.7 GiB, `audio_vae` 577 MB (F32, sha256 `52c59e67…26cb`). No `transformer_ref/`. |
| Prepared/quantised H3 | None on disk. Local runs use `--quantize 8` at load time (`--low-ram`); mlx-gen's measured footprint for 124 frames is 94.7 GB. |
| Other H3 files | `~/h3-weights/` (34 GiB): ComfyUI-format `minimax_h3_fl2va_pruned_fp8_scaled` (20 GiB) + `qwen3vl_32b…nvfp4_awq` TE (15 GiB). FL2VA only, ComfyUI layout — not mlx-gen-loadable, not Ref2VA. `models--Comfy-Org--MiniMax-H3` cache is empty (4 KB). |
| mlx-h3 | Not installed; no mlx-h3 checkpoints anywhere. |

## 2. Compatibility — what mlx-gen already has (v0.38.0, paths under `src/mflux/models/minimax_h3/`)

| Need for Ref2VA | Status | Where |
|---|---|---|
| Transformer architecture + loading + 8-bit quant | **Reusable as-is.** Ref2VA DiT has identical config and identical 638 tensor names | `model/h3_transformer/*`, `weights/h3_weight_mapping.py`, `weights/h3_weight_definition.py` |
| Qwen3-VL text + vision encoding, `<Picture i>: ` labels | Reusable; needs an `<Audio i>: ` label before the picture label | `variants/minimax_h3.py:_encode_presentation` (L429) |
| Video VAE encode of a conditioning image | Reusable (`_encode_keyframe_latents`, L479) | `model/h3_video_vae/h3_video_vae.py` |
| Audio VAE **encoder** | Implemented (`H3AudioVAE.encode`, L343; `H3AudioEncoder`, L154) and the checkpoint carries the 147 `encoder.*` tensors — but **never called and untested** | `model/h3_audio_vae/h3_audio_vae.py` |
| Packed layout with reference rows | Partial: `H3PackedLayout` already has `num_condition_audio_rows` and `build_row_timesteps` already pins conditioning audio at 1.0 — but `build_packed_sequence` hardcodes 0 audio conditioning rows and anchors the target at `text_len` | `latent_creator/h3_layout.py:226-318` |
| Scheduler / sampling loop | Reusable; already steps only non-conditioning rows | `variants/minimax_h3.py:226-251` |
| Model route selecting `transformer_ref` | **Missing** | `common/config/model_config.py`, `weights/h3_weight_definition.py:54` (`hf_subdir="transformer"`) |
| Reference-audio loading (32 kHz stereo, 2–15 s) | **Missing** | — |
| CLI flags | **Missing** (`--image-path` only) | `cli/minimax_h3_generate.py` |
| Docs | States "Ref2VA … not available yet" | `docs/minimax-h3.md:365` |

## Reference implementation (`appautomaton/mlx-h3` @ `adee6a3`)

- **Licence: none** (no LICENSE file, no licence field in README or pyproject) → all rights
  reserved. **Do not copy code**; use as a behavioural reference only. mlx-gen is MIT. The H3
  weights are under the *MiniMax H3 Community License*.
- `--ref-image` / `--ref-audio` are ordered references (≤9 images, ≤3 audio, ≤12 total);
  reference audio needs at least one image or video reference and is mutually exclusive with
  first/last keyframes. The CLI selects a separate DiT checkpoint (`dit_ref2va`) when any
  reference is present.
- Reference audio: ffmpeg → 32 kHz stereo, 2–15 s total → audio-VAE encoder → latent
  `[1, 32, 2, T]` → `2·T` rows.
- Layout (`layout.pack`): `[text | ref_img rows | ref_audio rows | target audio | target video]`.
  **References advance the timeline cursor** (an image by 1.0, audio by its `T`). The target
  audio and video start at that cursor, not at `text_len` as mlx-gen's keyframes do.
- Conditioning timesteps: ref image `max(t_v, 0.999)`, ref audio `max(t_a, 1.0)` — the same
  values mlx-gen already uses for keyframes and condition audio.
- Prompt presentation: `<Audio n>: ` precedes the `<Picture n>: ` label + vision block, in
  request order, then the prompt.
- Sync: target audio and video share the cursor origin and are denoised jointly (audio velocity
  rescaled to the video sigma grid).
- Validation: `tests/test_pipeline_ref2va.py` is **fully mocked** (FakeDiT, monkeypatched
  loaders). README calls Ref2VA "structurally compatible … passed local smoke validation, but
  experimental because its author has not yet declared Ref2VA support". There is no real-model
  parity evidence.

## 3. Model assets

| Component | Decision | Evidence |
|---|---|---|
| Text encoder (Qwen3-VL, 62.1 GiB bf16) | **REUSE** | `Ref2VA/text_encoder` 14/14 shards byte-identical (same blob id) to top-level |
| Video VAE (9.7 GiB) | **REUSE** | `Ref2VA/video_vae` blob identical |
| Audio VAE (0.56 GiB) | **REUSE** | `Ref2VA/audio_vae` blob identical; encoder tensors present |
| Tokenizer / processor / schedulers | **REUSE** | shared top-level files |
| Ref2VA transformer (`transformer_ref`) | **DOWNLOAD** 61.73 GiB bf16 (14 shards) | absent locally; all 14 weight shards differ from the base `transformer`; config and tensor names identical → **no CONVERT** (existing mapping and quantiser apply) |
| `Ref2VA/` bundle (134 GiB, original layout) | Do not download | its TE and VAEs duplicate the top level; use the diffusers `transformer_ref/` |
| ComfyUI `~/h3-weights` | Not applicable | FL2VA, ComfyUI format |
| Turbo LoRA | **UNKNOWN / none** | lightx2v Turbo is FL2VA-only; Ref2VA has no 8-step LoRA → base 50 steps |

**Extra disk:** about 62 GiB, or about 31–35 GiB more if a prepared 8-bit copy is kept (the bf16
copy can be dropped after that). 274 GiB is free. **Memory:** the same model size as today's
FL2VA run (94.7 GB at 124 frames with `--low-ram`), plus a small number of reference rows. A
5 s / 124-frame output stays inside the validated frame count.

## 4. Development readiness

- Fork `harkers/mlx-gen` created (did not exist). Clone `~/projects/mlx-gen`, with `origin` set to the fork and
  `upstream` set to `lpalbou/mlx-gen`. `gh` auth was fine (harkers, ssh).
- Worktree `~/projects/mlx-gen-worktrees/feat-minimax-h3-ref2va` on branch
  `feat/minimax-h3-ref2va` from `upstream/main` (`99fb94d`). It sits outside the clone because
  `.worktrees/` is not gitignored there. The `main` checkout is untouched and no PR has been opened.
- Conventions (CONTRIBUTING): `uv sync --extra dev`, `uv run ruff check src tests`,
  `uv run pytest`; tests for loading/quantisation/routing changes are required.
- **Baseline:** `MFLUX_FAKE_TOTAL_RAM_GIB=7 uv run pytest tests/minimax_h3
  tests/cli/test_mlx_gen_router.py -m "not slow and not high_memory_requirement"` →
  **299 passed, 1 skipped, 0 failed** at `99fb94d` (mlx 0.31.0 dev env, Python 3.13.13).

## 5. Minimal implementation plan (not started)

**Slice 1 — weights + conditioning**
- `common/config/model_config.py`: a `minimax-h3-ref2va` entry (MiniMaxAI/MiniMax-H3, 50 steps,
  shifts 12/3, transformer subdir `transformer_ref`, no turbo LoRA).
- `weights/h3_weight_definition.py`: transformer `hf_subdir` and download patterns follow the
  route (`transformer_ref/*` only for Ref2VA).
- `latent_creator/h3_layout.py`: `build_packed_sequence` accepts one ref image grid + ref audio
  latent length; emits ref rows, advances the cursor, sets `num_condition_audio_rows`; rejects
  refs combined with keyframe anchors.
- `variants/minimax_h3.py`: `ref_image_path` / `ref_audio_path`; presentation
  `<Audio 1>: <Picture 1>: [vision]` + prompt; ref image via `_encode_keyframe_latents`; ref
  audio load (ffmpeg, 32 kHz stereo, 2–15 s) → `audio_vae.encode` → rows.
- Tests: layout positions, row counts, tags and cursor advance (`test_h3_scheduler_layout.py`);
  `audio_vae.encode` shape `[1,32,2,T]` + encode→decode on tiny weights; presentation token order;
  route picks `transformer_ref`; tiny-pipeline Ref2VA pass leaves conditioning rows unchanged
  (`test_h3_pipeline_tiny.py`).
- Accept: all new tests plus the 299-test baseline green; ruff clean.

**Slice 2 — CLI**
- `cli/minimax_h3_generate.py` (and the `mlxgen generate` router): `--ref-image`, `--ref-audio`;
  reject them combined with `--image-path`; capabilities report a `ref2va` task.
- Tests: argument parsing and router tests (`tests/cli/test_mlx_gen_router.py`).
- Accept: `mlxgen generate --model minimax-h3-ref2va --ref-image … --ref-audio … --prompt …
  --frames 124` reaches preflight on fake RAM without loading weights.

**Slice 3 — end-to-end validation (needs the 62 GiB download, with approval)**
- One 124-frame generation: portrait still + a 10 s reference voice + one new dialogue line.
- Record peak memory and wall time.
- Judge voice resemblance against the reference voice (listening, plus a speaker-similarity
  score if available) and check dialogue accuracy (Whisper transcript vs script).
- Assess lip-sync on review frames, and publish to the tailnet review site.
- Accept: runs inside 128 GiB; transcript matches the line; the user judges voice and lips
  acceptable.

## 6. Risks

- **Ref2VA behaviour is unvalidated anywhere on MLX.** The only reference runtime has
  mocked tests and calls it experimental. The ref-audio row order relative to the ref image,
  and the label order, must be checked against MiniMax's own `scripts/readme/*ref2va*.sh` and
  `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` before Slice 1 is trusted.
- **Licence:** mlx-h3 has no licence → reimplement from behaviour only; never paste code.
- **Runtime:** no Turbo LoRA for Ref2VA → about 50 steps × ~45 s ≈ 35–40 min per 5 s clip at 544p
  (estimate from today's FL2VA step time; unmeasured).
- **Memory:** today's FL2VA footprint is already about 95 GB of 128; the Ref2VA transformer is the same
  size, so the headroom is the same — leave other GPU work (Ollama, mlx-broker models) unloaded.
- **Voice identity:** Ref2VA synthesises speech from a reference clip; whether it matches the
  reference speaker closely enough is unknown until Slice 3.
- **Dev env MLX 0.31.0 vs installed 0.32.3** — test on both before relying on parity.

## 7. Recommendation

**GO for Slices 1–2** (no download needed; everything except the transformer is reusable, and
the layout/timestep plumbing is half-built). **Gate Slice 3** on approval to download the
61.7 GiB `transformer_ref` and on first confirming the reference-order details against the
official MiniMax scripts.
