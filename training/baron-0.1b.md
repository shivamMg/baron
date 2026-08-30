# Baron-0.1B smoke-model training plan

Status: draft implementation specification  
Model ID: `baron-0.1b-base`  
Target hardware: one 8xH100 SXM/NVLink node  
Primary purpose: validate the complete Baron pretraining pipeline before the 400M scaling pilot and the 1.8B or 3B production run

## 1. Decision summary

Train a roughly 100.7M-parameter dense decoder-only Transformer for 1.000B tokens. It uses the same architectural family intended for the production model: pre-norm RMSNorm, RoPE, grouped-query attention (GQA), SwiGLU, tied embeddings, a 32K tokenizer, and a 4,096-token context.

This is an engineering smoke model, not a quality prototype. It should learn coherent English completion, but it is not expected to be a useful chatbot. Chat behavior will be evaluated later using the 400M pilot and production checkpoints after supervised fine-tuning and preference tuning.

The smoke run succeeds only if it proves all of the following:

- The corpus can be acquired, licensed, filtered, deduplicated, tokenized, shuffled, mixed, and replayed deterministically.
- An 8-GPU distributed job can train without NaNs, data stalls, skipped data, or unexplained loss spikes.
- Checkpoint save, durable copy, restore, and preemption recovery work.
- Evaluation, sample generation, metrics, and per-source validation work.
- The native checkpoint can be exported and loaded for standalone inference.
- Token counts and estimated compute can be reconciled with observed throughput closely enough to budget the next run.

Passing this run authorizes a 400M/10-30B-token scaling pilot. It does not by itself authorize the final 1.8B or 3B run.

## 2. Scope and non-goals

### In scope

- Base-model causal language modeling.
- A production-intent tokenizer that can be reused at larger scales.
- Four representative data streams: educational English web text, synthetic educational text, permissively licensed Python, and mathematical text.
- BF16 training through the production distributed and checkpoint paths.
- Native and Hugging Face-compatible checkpoint export.
- Base-model loss, generation, and lightweight downstream evaluation.

### Out of scope

- Producing a chat or instruction-following model.
- SFT, DPO, RLHF, tool use, or safety alignment.
- Proving the final data mixture is optimal.
- Using benchmark scores from a 100M model to choose between 1.8B and 3B.
- FP8 training. It may be benchmarked after the BF16 path passes.
- Long-context extension beyond 4,096 tokens.

## 3. Model configuration

| Setting                  |                                          Value |
| ------------------------ | ---------------------------------------------: |
| Architecture             |                 dense decoder-only Transformer |
| Layers                   |                                             12 |
| Model width              |                                            768 |
| Query heads              |                                             12 |
| KV heads                 |                                              4 |
| Head dimension           |                                             64 |
| MLP type                 |                                         SwiGLU |
| MLP intermediate width   |                                          2,048 |
| Normalization            |                               pre-norm RMSNorm |
| RMSNorm epsilon          |                                         `1e-5` |
| Position encoding        |                                           RoPE |
| RoPE base theta          |                                         10,000 |
| Native context length    |                                   4,096 tokens |
| Vocabulary size          |                                         32,768 |
| Input/output embeddings  |                                           tied |
| Linear-layer biases      |                                       disabled |
| Attention/MLP dropout    |                                            0.0 |
| Training dtype           |                                           BF16 |
| Parameter initialization | framework default, recorded in resolved config |

Approximate parameter count:

- Token embeddings: `32,768 x 768 = 25,165,824` parameters.
- Attention projections per layer: approximately 1,572,864 parameters.
- SwiGLU projections per layer: approximately 4,718,592 parameters.
- Twelve Transformer layers: approximately 75,497,472 parameters.
- Norm scales and small residual terms: less than 0.1M parameters.
- Total with tied output embeddings: approximately **100.7M parameters**.

The 12:4 attention layout is intentionally grouped rather than conventional multi-head attention. It tests the GQA kernels and checkpoint format that the larger inference-oriented models will use.

## 4. Tokenizer

The tokenizer is a production artifact. Do not substitute an existing model's tokenizer merely to make the smoke run start sooner.

### Specification

- Hugging Face Tokenizers byte-level BPE.
- Vocabulary size: 32,768 including reserved tokens.
- No unknown-token path; every byte sequence must round-trip.
- Preserve case, whitespace, indentation, and code punctuation.
- Convert invalid byte sequences to valid UTF-8 and normalize line endings. Avoid lossy text normalization.
- Add one EOS token between packed documents. Do not add BOS by default.
- Reserve tokens for EOS, padding, system, user, assistant, tool calls, tool results, and future fill-in-the-middle use. The final token strings and IDs must be frozen in `tokenizer/special_tokens.json` before pretraining.

### Training sample

Build a deterministic, document-level sample of 10-20GB of UTF-8 text using the same 75/15/5/5 source proportions as the smoke corpus. Sample before tokenization, cap very large documents, and shuffle by a seeded content hash.

### Acceptance checks

- Encode/decode round-trips arbitrary UTF-8, whitespace, and Python fixtures.
- No special token is produced accidentally from ordinary source text.
- No held-out document used for tokenizer evaluation enters tokenizer training.
- Report bytes/token by source, token-length percentiles, and special-token frequency.
- English held-out text should average no worse than 4.2 UTF-8 bytes/token. If it misses this gate, inspect the sample and tokenizer before training.
- Save `tokenizer.json`, tokenizer configuration, special-token mapping, training manifest, source revision IDs, code revision, and SHA-256 hashes.

The same frozen tokenizer should be used by the 400M pilot and production model unless the smoke run identifies a concrete defect. Changing it later invalidates direct checkpoint continuation and complicates comparisons.

## 5. Data plan

### Training mixture

The target is 3,815 optimizer steps at 262,144 tokens/step, or exactly 1,000,079,360 consumed tokens.

| Stream | Public repository and configuration | Weight | Train tokens |
| --- | --- | --: | --: |
| English educational web | `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` | 75% | 750,059,520 |
| Synthetic educational text | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | 15% | 150,011,904 |
| Educational Python | `HuggingFaceTB/smollm-corpus`, `python-edu` | 5% | 50,003,968 |
| Mathematical text | `HuggingFaceTB/finemath`, `finemath-4plus` | 5% | 50,003,968 |

Use exact dataset repositories, configurations, and immutable revisions in a checked-in data manifest. The names above identify the intended corpus families, not permission to consume a mutable `main` revision.

Do not add DCLM or another broad Common Crawl derivative to this smoke mixture. It would substantially overlap FineWeb and add deduplication uncertainty without helping this pipeline test.

### Validation data

Reserve 512 packed sequences (2,097,152 tokens) from each source before train selection, for 8,388,608 validation tokens in total. Maintain both:

- One fixed combined validation set using the training mixture weights.
- Four fixed source-specific validation sets for diagnosing regressions.

Also maintain a small, separately sourced English validation set for detecting overfitting to the selected corpus family. Never use benchmark test data for training, tokenizer fitting, or threshold tuning.

### Licensing and provenance gate

Before downloading content, record for every source:

- Repository and configuration name.
- Immutable dataset revision/commit.
- Dataset card and source URL.
- Dataset-level license and any per-document license metadata.
- Allowed-use decision and reviewer/date.
- Download tool version and acquisition timestamp.
- Document identifier, source identifier, and content hash.

For code, apply an explicit approved-license allowlist. Exclude files with missing, unknown, or disallowed license metadata rather than treating them as permissive. Python-Edu stores Software Heritage blob IDs and requires fetching content through the documented Software Heritage path; preserve its underlying file-level license and attribution records outside the token shards. FineMath is published under ODC-By 1.0 and remains subject to Common Crawl's terms of use. Review the pinned dataset cards because corpus licensing and access methods can change. Dataset documentation is evidence, not a substitute for legal review.

If Python-Edu cannot pass the license gate, replace its 5% quota with FineWeb-Edu-Dedup for this smoke run and record the deviation. Do not delay the pipeline validation by silently weakening the license policy.

### Processing pipeline

Use DataTrove or equivalent structured stages. Each stage writes counts and a manifest so that a failed job can resume without rebuilding accepted outputs.

1. Acquire immutable source revisions into durable raw storage.
2. Parse records and retain source IDs and provenance.
3. Validate UTF-8, normalize line endings, and remove binary or corrupt text.
4. Apply source-specific quality filters and English-language filtering where appropriate. Do not apply an English-only filter to code.
5. Remove exact duplicates globally by normalized document hash.
6. Apply MinHash or another documented near-duplicate pass across the selected sources, not only within each source.
7. Remove repeated boilerplate and obvious spam.
8. Apply documented PII detection/removal. Record aggregate removal counts but never log detected PII content.
9. Split by document hash into tokenizer sample, train, validation, and audit sets before tokenization.
10. Tokenize with the frozen tokenizer, append EOS at document boundaries, and pack without padding into fixed 4,096-token sequences.
11. Write sharded token data plus index and checksums. Use `uint32`; 32K token IDs do not justify a format that limits future vocabulary changes.
12. Construct a deterministic interleaved manifest that enforces the exact source quotas above and supports replay from any optimizer step.

Target 25-50M tokens per final shard. Randomize shard and record order using a recorded seed, while keeping validation order fixed. A document may be split across adjacent sequences, but train and validation must never share a document.

### Data acceptance checks

- Source counts reconcile at every processing stage.
- Requested and emitted token counts match exactly.
- Duplicate train/validation document hashes: zero.
- Every final shard passes checksum and token-ID range validation.
- At least 1,000 decoded random samples receive a manual spot check, stratified by source.
- The loader reproduces the same first 100 sequence hashes after restart and when changing worker count.
- No sequence consists mostly of padding, boilerplate, replacement characters, or a single repeated token.

## 6. Training configuration

| Setting               |                                               Value |
| --------------------- | --------------------------------------------------: |
| Objective             |                            next-token cross entropy |
| Train tokens          |                                       1,000,079,360 |
| Sequence length       |                                               4,096 |
| Microbatch per GPU    |                          2 sequences / 8,192 tokens |
| GPUs                  |                                                   8 |
| Gradient accumulation |                                        4 microsteps |
| Global batch          |                       64 sequences / 262,144 tokens |
| Optimizer steps       |                                               3,815 |
| Optimizer             |                                               AdamW |
| Adam beta 1           |                                                 0.9 |
| Adam beta 2           |                                                0.95 |
| Adam epsilon          |                                              `1e-8` |
| Peak learning rate    |                                              `6e-4` |
| Warmup                |                          76 steps, approximately 2% |
| Decay                 |                                    cosine to `6e-5` |
| Weight decay          |          0.1 on matrix weights; exclude norm scales |
| Gradient clipping     |                                     global norm 1.0 |
| Z-loss                | `1e-4`, if supported consistently by train and eval |
| Dropout               |                                                 0.0 |
| Model seed            |                                                1337 |
| Data seed             |                                                1338 |
| Precision             |   BF16 parameters/activations; FP32 optimizer state |

Schedule the learning rate by consumed non-padding tokens, not wall clock. The resolved run configuration must state whether the tied embedding matrix receives weight decay and whether cross-entropy statistics include or exclude z-loss.

### Distributed configuration

- PyTorch distributed with NCCL.
- OLMo-core as the preferred training framework, pinned to a commit.
- FSDP2/full sharding so the smoke run exercises the production checkpoint and resharding path, even though this model fits easily on one GPU.
- PyTorch SDPA Flash Attention or a pinned FlashAttention implementation.
- Activation checkpointing at Transformer-block boundaries, matching the intended production path.
- No tensor or pipeline parallelism on this single-node run.
- Enable TF32 for permitted FP32 matrix operations.
- Start with `torch.compile` disabled. Enable it only after the eager BF16 canary passes, then record compile time and throughput delta.

Pin the container image digest, CUDA, NCCL, Python, PyTorch, OLMo-core, tokenizer, data-processing, and attention-kernel versions. A loose dependency range is not sufficient for a multi-week production run.

## 7. Execution stages

Do not submit the full 1B-token job first. Run these stages in order.

### Stage A: unit and data canary

- Run one forward/backward optimizer step on CPU where supported and one GPU.
- Verify shifted labels, causal masking, EOS handling, loss masking, and token accounting with a hand-constructed batch.
- Decode loader output and compare it with the source records.
- Confirm all parameters expected to train receive finite gradients.

Pass condition: deterministic fixtures and token counts pass with no NaNs.

### Stage B: deliberate overfit

- Use one GPU and 1-2M tokens from a disposable train-only slice.
- Repeat the slice until loss falls below 1.0 or generation clearly memorizes the fixture.
- Save, reload, and continue from a checkpoint midway through the test.

Pass condition: the model can overfit, restore, and continue. Failure usually indicates a masking, label-shift, optimizer, or checkpoint defect.

### Stage C: distributed canary

- Train all eight GPUs for 50 optimizer steps.
- Save at step 10, terminate the process after the durable copy completes, and resume from that checkpoint.
- Compare resumed loss with an uninterrupted reference around the same step.
- Exercise evaluation, sample generation, and native-to-HF export.

Pass condition: all ranks consume unique expected data, restart loses no more than one incomplete step, and resumed loss differs by less than 1% from the reference trajectory. Bitwise identity is not required for distributed BF16.

### Stage D: full smoke run

- Train through step 3,815.
- Evaluate every 250 steps and at the final checkpoint.
- Generate a fixed prompt suite every 250 steps.
- Save a normal checkpoint every 250 steps, plus milestone checkpoints at approximately 25%, 50%, 75%, and 100% of tokens.
- Retain the last three rolling checkpoints and every milestone checkpoint.

Pass condition: all acceptance gates in Section 10 pass.

## 8. Storage and checkpointing

The AML bootstrap mounts durable Blob storage at `/mnt/baron-training`. BlobFuse is not fully POSIX-compatible, so training must not write a live checkpoint directly into that mount.

Use this flow:

1. Write each checkpoint to node-local NVMe/scratch under a temporary name.
2. Flush and close all files, write checksums, and atomically mark the local checkpoint complete.
3. Copy the completed directory to `/mnt/baron-training/checkpoints/baron-0.1b-base/<run-id>/<step>/`.
4. Verify durable file sizes and checksums.
5. Write a small `COMPLETE.json` marker last. A restore job may only select a checkpoint with a valid marker and checksums.

Each checkpoint must include:

- Model, optimizer, scheduler, gradient-scaler state if any, and RNG state.
- Global optimizer step, consumed-token count, epoch-independent data cursor, and source-mixture cursor.
- Resolved model/training config and all software revision IDs.
- Tokenizer files and hashes.
- Data-manifest ID and shard checksums.
- Metrics immediately before the save.

Copy logs, resolved configs, manifests, evaluation output, and exported final artifacts under `/mnt/baron-training/runs/baron-0.1b-base/<run-id>/`.

## 9. Metrics and evaluation

### Runtime metrics

Log at least once per optimizer step:

- Total and cross-entropy loss, z-loss if enabled, and learning rate.
- Gradient norm and clipping frequency.
- Tokens/second per GPU and aggregate.
- Step, data-loader, forward, backward, optimizer, evaluation, and checkpoint durations.
- GPU utilization, allocated/reserved memory, temperature, and power.
- Per-source token counts and cumulative mixture percentages.
- MFU estimate with its formula and hardware peak assumption.

A 100M model will underutilize eight H100s; low MFU alone is not a failure. Data stalls, rank imbalance, unstable throughput, or unexplained idle periods are failures because they would become expensive at production scale.

### Validation

At each evaluation point report combined and per-source token-weighted negative log likelihood. Perplexity may also be reported, but only alongside the exact tokenizer and token-weighting method.

Use a fixed base-model prompt suite covering:

- English continuation and paragraph completion.
- Basic factual syntax without treating factual accuracy as a smoke gate.
- Summarization-shaped continuation.
- Python completion.
- Simple arithmetic and mathematical prose.
- Repetition and EOS behavior.

Decode with greedy generation and at least one fixed seeded sampling setting. Do not use a chat template or score conversational helpfulness for this base checkpoint.

Run a small, fixed `lm-evaluation-harness` subset such as HellaSwag, PIQA, ARC-Easy, and LAMBADA only to validate the harness and establish a baseline. Scores at this scale are diagnostic, not model-selection evidence.

### Export verification

- Export final weights and configuration to a Transformers-compatible format.
- Load native and exported checkpoints on one GPU.
- On fixed token fixtures, compare logits within documented BF16 tolerances (`rtol=0.02`, `atol=0.02` initially).
- Confirm tied embeddings remain tied logically after export.
- Generate with both runtimes and investigate any token divergence before the first sampling decision.

## 10. Full-run acceptance gates

The smoke model passes only when all required gates are recorded in a signed-off run report.

### Required

- No NaN or infinite loss, gradient, parameter, or optimizer state.
- No unexplained loss spike greater than 20% that persists for three steps.
- Combined held-out cross entropy improves by at least 30% relative to the first post-warmup evaluation.
- No source-specific validation loss worsens for three consecutive evaluations while combined loss improves without an understood mixture explanation.
- Actual source proportions finish within 0.1 percentage point of their targets.
- Consumed-token count exactly matches 1,000,079,360.
- At least one forced termination restores successfully from durable storage.
- Native/HF export verification passes.
- Fixed prompts avoid persistent empty output, immediate EOS, or repeated-token collapse at the final checkpoint.
- Metrics contain no unexplained gap longer than two optimizer steps.
- Final data, tokenizer, software, config, and checkpoint manifests are complete and checksum-valid.

### Investigate but do not automatically fail

- Low benchmark accuracy expected from a 100M model.
- Low 8-GPU MFU caused by the deliberately small model.
- Isolated transient loss spikes attributable to a recorded data batch.
- Modest English quality, provided generation improves and is non-degenerate.

## 11. Time and resource estimate

The nominal Transformer training compute is:

`6 x 100.7M parameters x 1.000B tokens`, or approximately `6.04e17` FLOPs.

This is tiny relative to eight H100s, so communication, kernel launch, data, evaluation, and startup overhead dominate. Plan for:

- Tokenizer and 1B-token corpus preparation: 4-24 hours depending on source download locality and preprocessing parallelism.
- Stages A-C: 1-3 hours including deliberate restart and export checks.
- Stage D training: approximately 1-4 hours after data is local and cached.
- End-to-end reservation: one 8-hour GPU window after prepared data is ready, with an additional window available for one corrected rerun.

Measure actual steady-state tokens/second and component timings. Do not use this small model's MFU to extrapolate final duration directly; use the 400M pilot and a short exact-shape 1.8B/3B benchmark for production estimates.

## 12. Failure policy

Stop immediately for NaNs, corrupt checkpoints, train/validation leakage, unexpected source mixture, missing provenance, persistent rank imbalance, or a loader that cannot replay after restart.

For each stopped run:

1. Preserve the resolved config, logs, last good checkpoint, failing batch identifiers, and environment manifest.
2. Classify the cause as data, model, optimizer, distributed runtime, storage, export, or infrastructure.
3. Add the smallest regression test that reproduces the defect.
4. Resume only from a checkpoint known to precede corrupted state.
5. Assign a new run ID when code, data, tokenizer, or hyperparameters change.

Do not silently skip a bad batch. Quarantine it with provenance and determine whether the filter or loader needs correction.

## 13. Deliverables

A successful run produces:

- Frozen tokenizer and tokenizer-training manifest.
- Raw-to-token data lineage manifest with immutable revisions and licenses.
- Checksummed training and validation shard manifests.
- Resolved model, optimizer, scheduler, distributed, and environment configs.
- Native final and milestone checkpoints.
- Transformers-compatible final checkpoint.
- Runtime metrics, per-source validation, generation samples, and harness output.
- Restart comparison and export-equivalence reports.
- A run report containing failures, fixes, achieved throughput, observed MFU, peak memory, total GPU-hours, and acceptance-gate results.
- A recommendation to proceed, repeat, or revise before the 400M pilot.

## 14. Scale-up decision

Proceed to the 400M pilot only after every required gate passes. Carry forward the tokenizer, data schema, manifest format, checkpoint format, monitoring, evaluation prompts, and export path unchanged where possible.

The next pilot should use approximately 400M parameters and 10-30B tokens to measure learning curves, data-mixture effects, throughput, and early capability. Only after that pilot should the project lock the 1.8B inference-first or 3B quality-first production configuration and its final token budget.

## 15. Implementation checklist

- [ ] Pin the training container and all software revisions.
- [ ] Add the approved data-source/license manifest.
- [ ] Implement and test tokenizer training.
- [ ] Implement resumable data processing and global deduplication.
- [ ] Build immutable train and validation token manifests.
- [ ] Implement the 100.7M model config in the selected trainer.
- [ ] Add unit fixtures for masks, labels, packing, and token accounting.
- [ ] Add durable two-phase checkpoint copy and restore selection.
- [ ] Add combined and per-source evaluation.
- [ ] Add fixed-prompt generation and HF export verification.
- [ ] Run Stages A, B, and C.
- [ ] Review canary evidence before approving Stage D.
- [ ] Run Stage D and complete the acceptance report.
- [ ] Decide whether to proceed to the 400M pilot.

## 16. References

- OLMo-core training framework: <https://github.com/allenai/OLMo-core>
- SmolLM-Corpus dataset card: <https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus>
- FineMath dataset card: <https://huggingface.co/datasets/HuggingFaceTB/finemath>
- FineWeb-Edu paper: <https://arxiv.org/abs/2406.17557>
- SmolLM2 training report: <https://arxiv.org/abs/2502.02737>
- OLMo 2 technical report: <https://arxiv.org/abs/2501.00656>
- DataTrove processing framework: <https://github.com/huggingface/datatrove>
- Chinchilla scaling analysis: <https://arxiv.org/abs/2203.15556>

Before execution, archive the exact dataset cards and software documentation used for the pinned revisions. Web pages and mutable repository branches are not reproducibility records.
