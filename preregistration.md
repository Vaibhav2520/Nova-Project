# Preregistration: Cross-Execution Consensus for NOVA

Status: committed before any implementation. Commit hash and UTC timestamp are taken from the git commit that first adds this file.
Author: Vaibhav Sulakhe

## 1. Primitive
For demonstrations (x_k, y_k), k = 1..K: h_k = Enc(x_k, y_k). R[k,j] = NLL(Exec(x_j, h_k), y_j) for k != j.
Trust w = softmax_k(-mean_{j!=k} R[k,j] / tau_w). Reliability v = softmax_j(-min_{k!=j} R[k,j] / tau_v). z_0 = sum_k w_k h_k.
Refinement: z_{t+1} = z_t + alpha * clip(Upd(z_t, sum_j v_j Err(y_j, Exec(x_j, z_t)))).
Halt when rho_t = sum_j v_j NLL(Exec(x_j, z_t), y_j) < eps, when rho stops improving by delta, or at t = T_max.
Answer: Exec(x_q, z*), where z* is the best-so-far rule.

## 2. Mechanism
Rule formation is separated from rule reuse, and each candidate rule is verified by execution on the other demonstrations
before it is trusted. Aggregation is over sets, so it is order-invariant. Effort, confidence and continual plasticity all
follow the same verifiable fit signal rho.

## 3. Main hypothesis
Under equal parameter count (+/-5%), optimizer-update budget and training-example budget, NOVA achieves a higher GEI than a
causal Transformer and than a compute-matched looped Transformer (Transformer-CM, FLOPs within 10%) on compositional and
structural OOD splits, because cross-execution consensus verifies rules before reuse.

## 4. Predictions that differ from the Transformer
- P1. With one of K >= 3 demonstrations corrupted, NOVA's accuracy drops less.
- P2. Permuting demonstrations leaves NOVA's outputs identical within 1e-5, while the Transformer's change (design property).
- P3. Refinement steps correlate positively with composition depth (Spearman > 0.3), and easy tasks use measurably fewer executor calls and less wall-clock time.
- P4. NOVA's confidence has lower ECE than temperature-scaled softmax confidence on OOD splits.
- P5. NOVA's accuracy rises more steeply with K than the Transformer's.
- P6. In the continual stream, residual-gated plasticity forgets less than ungated online updates and a matched lower learning rate, at the same replay cap.

## 5. Falsifiers
- F1. The paired bootstrap 95% CI of the GEI difference includes zero on two or more of the three static domains, or the sign flips across the three seeds.
- F2. Ablated NOVA, the more-parameters control or the shuffled-R control matches full NOVA within the CI.
- F3. A compute-matched fixed-T NOVA matches adaptive NOVA, which falsifies the adaptive-compute claim.
- F4. P1 fails: no robustness advantage under one corrupted demonstration.
- F5. NOVA's advantage disappears against Transformer-CM, which would mean the gain was bought with compute.

## 6. Expected failure regime
K = 2, where a corrupted demonstration cannot be identified. Rules that no single demonstration pins down. Very long inputs
needing global operations such as sort. Count outputs never seen in training. Duplicated demonstrations, which gain extra weight.

## 7. Closest prior mechanisms and departure
CNP mean pooling becomes execution-verified weighting. Latent Program Network test-time gradient search becomes a learned
forward update. Iterative amortized inference gradient inputs become execution residuals. DSAC, execution-guided synthesis
and CodeT verify hypotheses or discrete programs; CEC verifies continuous latent rules with a learned executor, differentiably.
ACT and Universal Transformer learned halting becomes halting on a verifiable fit residual. NTM external memory is not used.

## 8. Fixed budget
- About 1.0M parameters, +/-5% for every learned model. Transformer-CM FLOPs within 10% of NOVA's.
- 10,000 updates, 32 tasks per update. Learning-curve budgets 156 to 10,000 updates, log2-spaced.
- Seeds 0, 1, 2. Experiment B on sequences x 3 seeds plus graphs x 1 seed.
- Tuning: 6 trials per model at 1,500 updates on iid_val, including every evaluation-time knob.
- Replay cap: 512 tasks, <= 10% per batch. GEI penalties per technical design section 8.1.

## 9. Decisions that may change after the pilot
May change: learning rate and warm-up within the tuning budget; tau, eps, delta and T_max <= 8 chosen on iid_val; executor
read option A or B; micro-batch size; the four-step reduction ladder, applied uniformly.
May not change: splits, seeds, metrics, decision rule, falsifiers, baselines including Transformer-CM, parameter and FLOP tolerances.
