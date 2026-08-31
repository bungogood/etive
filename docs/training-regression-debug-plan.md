# Training Regression Debug Plan

> Historical investigation record, completed on 2026-08-31. See
> `benchmarks/training-regression/README.md` for the resulting report and data.
> The plan below is preserved as executed; unchecked items were not completed
> and are not current instructions.

## Objective

Determine why the continuously trained Othello network regressed after generation 4, correct the training system, and establish a repeatable process for producing a strong engine.

This investigation treats playing strength against fixed checkpoints as the primary outcome. Training loss is supporting evidence, not proof of improvement.

## Preserved Experiment

- Configuration: historical deployed version of `experiments/weekend-4x64-256.toml`; the current file writes corrected runs to a new output directory
- Remote output: `/home/jonathan/projects/etive-rootcause-selector/checkpoints/weekend-4x64-256`
- Preserved state at investigation start: paused after generation 12, with no pending self-play manifest
- Model: 4 residual blocks, 64 channels, 8 normalization groups
- Self-play: 4,096 games per generation, 256 simulations per move
- Training: batch size 128, replay reuse 4, replay capacity 4,000,000 positions
- Evaluation: 400 paired games every four generations
- Known-good checkpoint: generation 4

During the investigation, this output directory was not resumed or cleaned.

## Current Evidence

### Arena Results

| Contender | Baseline | Seed | Wins | Losses | Draws | Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Generation 4 | Generation 0 | 4242 | 337 | 49 | 14 | 86.0% |
| Generation 8 | Generation 4 | 4242 | 105 | 270 | 25 | 29.4% |
| Generation 4 | Generation 8 | 4242 | 270 | 105 | 25 | 70.6% |
| Generation 8 | Generation 4 | 9001 | 105 | 282 | 13 | 27.9% |
| Generation 12 | Generation 4 | 4242 | 41 | 353 | 6 | 11.0% |
| Generation 12 | Generation 4 | 9001 | 38 | 354 | 8 | 10.5% |

Reversing generation 4 and generation 8 produced exactly reversed results for the same opening seed. An alternate seed reproduced the regression. The arena evaluator is therefore unlikely to be the primary fault.

Generation 12's 53.6% evaluation score was against generation 8, not generation 4. The current ordering is approximately generation 4 >> generation 12 > generation 8.

### Loss Evidence

- Training policy KL changed only from approximately 2.31 to 2.23 through generation 12.
- Policy cross-entropy remained around 4.108, close to the uniform 65-action baseline of `ln(65) = 4.174`.
- Value MSE remained around 0.95-0.96, close to the zero-prediction baseline for outcomes in `{-1, 0, 1}`.
- Training and validation losses remained close, so conventional train-set overfitting is not the leading explanation.

## Working Hypotheses

Investigate these in order. Do not tune the long-run configuration until the earlier hypotheses are resolved.

1. Real replay batches are not producing effective policy or value updates despite the synthetic fixed-batch overfit test.
2. Policy targets, outcome perspective, action symmetry, or pass indexing are incorrect in persisted self-play data.
3. CPU and CUDA training diverge materially for real replay batches.
4. The learner is dominated by stale early replay data during buffer warmup.
5. Pure latest-network self-play amplifies an early learner regression.
6. The model, optimizer, or learning rate is unsuitable for the diversity of a 4,096-game generation.

## Phase 1: Preserve Evidence

- [x] Record hashes and sizes for configuration, state, metrics, model checkpoints, and optimizer checkpoints.
- [x] Record the local source revision and hashes of the deployed source and binary.
- [x] Preserve generations 0, 4, 8, and 12 as permanent diagnostic anchors.
- [x] Save the independent arena matrix as machine-readable CSV.
- [x] Confirm that replay files required for frozen-data experiments are present.

Exit gate: the experiment can be reconstructed without relying on terminal history.

## Phase 2: Checkpoint Diagnostics

Add a deterministic diagnostic command that evaluates a checkpoint on frozen replay data and emits CSV or JSON with:

- Policy cross-entropy, target entropy, and KL
- Probability mass assigned to legal actions
- Top-1 and top-k agreement with the MCTS target
- Predicted policy entropy
- Value MSE and mean absolute error
- Value sign accuracy and outcome correlation
- Predicted value mean and standard deviation
- Results divided into opening, middle, and endgame positions

Run it for generations 0, 4, 8, and 12 on exactly the same replay rows.

Exit gate: identify which outputs changed between the strong and regressed checkpoints.

Status: complete. Generations 4, 8, and 12 have effectively constant value output and nearly position-independent policy output. FP16 and FP32 diagnostics agree.

## Phase 3: Frozen-Replay Training

Remove self-play dynamics from the experiment:

1. Load generation 4 and its optimizer.
2. Freeze real training and validation rows.
3. Train independent copies for 100, 500, 1,000, and 5,000 steps.
4. Record diagnostics and checkpoints at each interval.
5. Arena-test each checkpoint against the unchanged generation 4.

Run at least these learning rates:

| Trial | Learning rate |
| --- | ---: |
| A | 0.001 |
| B | 0.0003 |
| C | 0.0001 |

Expected behavior: training policy KL falls clearly, validation metrics do not diverge sharply, and arena strength does not collapse.

Interpretation:

- Loss fails to decline: investigate gradients, optimizer state, target construction, or model capacity.
- Loss declines but strength collapses: investigate target semantics and mismatch between the objective and search strength.
- Frozen training succeeds: investigate replay age and self-play feedback dynamics.

Status: root cause isolated. Seed 8 at approximately `1e-3` collapses the network, while the same seed at `3e-4` learns normally. After 5,000 steps at `3e-4`, policy KL reached 0.262 and the checkpoint beat generation 0 by 398-2.

## Phase 4: CPU/CUDA Agreement

Using the same checkpoint, optimizer state, replay batch, random seed, and symmetry:

- [ ] Compare initial CPU and CUDA outputs.
- [ ] Compare policy and value losses.
- [ ] Compare gradient and parameter-update norms.
- [ ] Compare outputs after 1, 10, and 100 optimizer steps.

Small floating-point differences are expected. Qualitatively different loss trajectories or large output divergence indicate a backend problem.

Exit gate: CPU and CUDA agree within a documented tolerance.

Status: complete for the collapse-producing first update. CUDA and Flex losses agree within approximately `5e-5`, ruling out a CUDA-specific failure as the leading cause.

## Phase 5: Target Audit

Validate persisted replay and freshly generated samples:

- [ ] Every policy is finite, non-negative, and sums to one.
- [ ] Illegal actions have zero target probability.
- [ ] Pass has probability only when no board move is legal.
- [ ] All eight symmetries transform positions and policies consistently.
- [ ] Outcome is from the encoded side-to-move perspective.
- [ ] Late-game outcomes agree with exact terminal results.
- [ ] Root visit counts produce the persisted policy exactly.
- [ ] Tactical near-terminal positions have the expected MCTS and value signs.

Exit gate: no target or perspective invariant is violated.

## Phase 6: Replay Ablation

Starting from generation 4, run short experiments that differ only in replay policy:

| Trial | Replay policy |
| --- | --- |
| A | Latest generation only |
| B | Last four generations |
| C | Growing window capped at four million positions |
| D | Recency-weighted rolling replay |

Evaluate every checkpoint against fixed generations 0 and 4. Do not rely only on adjacent-checkpoint matches.

Exit gate: choose the smallest replay policy that provides stable improvement and validation diversity.

## Phase 7: Learner and Champion Separation

Implement AlphaGo Zero-style separation:

- Self-play uses the current champion.
- The learner and optimizer continue after a failed evaluation.
- Failed evaluation does not reset learner weights or optimizer state.
- Failed evaluation does not replace the self-play champion.
- Promotion requires more than 55% over 400 paired games.
- State records learner, champion, and last-evaluated generations separately.

This prevents a severely regressed learner from generating the next replay data while still allowing several modest training updates to accumulate.

Exit gate: pause/resume tests preserve all three generation identities, and rejected learners continue training without affecting self-play.

Status: implemented. The corrected four-hour preflight promoted generation 4 after a 400-0 arena result against generation 0; subsequent self-play uses generation 4 while the learner continues independently.

## Phase 8: Reporting

Generate a report for every experiment containing:

- Configuration and source revisions
- Self-play positions, evaluations per second, and elapsed time
- Training steps per second and learning rate
- Training and validation policy KL
- Training and validation value loss
- Legal policy mass and value correlation
- Replay size and age distribution
- Arena scores with confidence intervals
- Strength against fixed anchor checkpoints
- Approximate Elo trajectory

Store raw report data as CSV and render charts from that data. Reports must remain reproducible without parsing terminal logs.

## Preflight Gates for Another Long Run

- [x] Frozen real replay produces a clear policy-KL reduction.
- [x] Value predictions develop meaningful variance and outcome correlation.
- [x] CPU and CUDA update trajectories agree for the collapse-producing first update.
- [ ] No target invariant fails.
- [x] A trained checkpoint remains above 45% against its starting checkpoint.
- [ ] Two short self-play trials improve against fixed anchors.
- [ ] Learner/champion resume behavior is tested.
- [ ] Automated reports and charts are generated successfully.

After these pass, run a two-to-four-hour trial before another 72-hour experiment.
