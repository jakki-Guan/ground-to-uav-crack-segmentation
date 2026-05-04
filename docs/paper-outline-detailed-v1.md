# Detailed Paper Outline v1

**Target venue:** `Automation in Construction` first, `Drones` second  
**Source plan:** [paper-plan-v4.md](./paper-plan-v4.md)  
**Frozen result references:** [baseline-results.md](./baseline-results.md), [research-plan.md](./research-plan.md), [results/report_assets/final_comparison/summary.md](../results/report_assets/final_comparison/summary.md)

## 0. Working Framing

**One-sentence framing**

This paper provides a deployment-oriented diagnostic workflow for UAV pavement crack segmentation: quantify source-only transfer failure, diagnose structured false positives, mitigate them with low-cost `GT`-filtered target-background / hard-negative exposure, and determine how little target annotation is needed to approach target-domain performance.

**Working thesis**

Ground-to-UAV crack segmentation is best framed as a deployment diagnosis problem rather than a new-architecture problem: source-only transfer fails, false positives are structured and interpretable, lightweight `GT`-filtered target-background / hard-negative mitigation helps, and modest UAV annotation closes most of the remaining gap.

## 1. Title And Abstract

### 1.1 Candidate titles

1. Ground-to-UAV Pavement Crack Segmentation Under Domain Shift: Deployment Failure Diagnosis, Hard-Negative Mitigation, and Few-Shot Recovery
2. Diagnosing and Mitigating Ground-to-UAV Domain Shift for Pavement Crack Segmentation
3. Ground-to-UAV Crack Segmentation Under Domain Shift: Failure Diagnosis, Hard-Negative Mitigation, and Few-Shot Recovery
4. From Ground Imagery to UAV Deployment: A Diagnostic Study of Crack Segmentation Failure and Recovery

### 1.2 Abstract structure

1. Problem and motivation:
Ground-trained crack segmentation models are attractive for UAV-assisted pavement inspection, but cross-domain deployment reliability remains unclear.

2. Gap:
Most prior work emphasizes model design or in-domain benchmarks, while the deployment failure modes and practical low-cost mitigation paths are less clearly characterized.

3. Method summary:
Use `Crack500` as source training data and `UAV_Crack_Segmentation_Kaggle` as target evaluation data. Measure source-only transfer, audit false positives, test `GT`-filtered target-background / hard-negative mitigation (`B1`), and evaluate few-shot target fine-tuning (`B2`).

4. Key numbers to report:
- `SegFormer-B2` source-only `IoU 0.1442`
- promoted `SegFormer-B2 B1` `IoU 0.3775`
- promoted `DeepLabV3+ B1` `IoU 0.3860`
- `SegFormer-B2 fs20` `IoU 0.5686`
- `SegFormer-B2` upper bound `IoU 0.5879`

5. Interpretation:
False positives are structured rather than random; `GT`-filtered target-background / hard-negative mitigation helps, but bank-quality effects are model-dependent; limited UAV annotation closes most of the remaining gap.

6. Contribution statement:
This work provides a deployment-oriented diagnostic workflow and practical adaptation roadmap for UAV pavement inspection.

## 2. Introduction

### 2.1 Opening paragraph

- Motivate UAV-assisted pavement inspection as a practical setting where rapid coverage, oblique viewpoints, and safer inspection workflows matter.
- State that segmentation models trained on common ground-view crack datasets may be reused for UAV deployment because target annotations are expensive.
- Establish the central risk: performance can collapse under viewpoint, scale, texture, and illumination shift.

### 2.2 Problem paragraph

- Position the problem as cross-domain deployment rather than in-domain benchmarking.
- Emphasize that poor deployment is not only a matter of lower recall; structured high-confidence false positives also matter because they can create misleading defect maps.

### 2.3 Opportunity paragraph

- Explain why the paper focuses on low-cost mitigation rather than full-scale domain adaptation.
- Introduce two practical levers:
  - `GT`-filtered target-background / hard-negative mitigation (`B1`)
  - limited-label few-shot recovery (`B2`)

### 2.4 Paper framing paragraph

- State that the paper does not propose a new architecture.
- Position the work as a controlled diagnostic study with practical construction-inspection implications.

### 2.5 Contribution paragraph

Suggested final bulletized contributions:

1. A controlled ground-to-UAV crack segmentation benchmark showing a severe zero-shot deployment gap.
2. A false-positive diagnosis framework combining qualitative audit and nuisance taxonomy.
3. An evaluation of `GT`-filtered target-background / hard-negative mitigation that improves deployment performance without directly optimizing on target-domain positive crack pixels.
4. A few-shot supervision study showing that modest UAV annotation closes most of the remaining gap.
5. A claim-bounded analysis showing that mined-bank value is model-dependent rather than universally stronger than matched random background.

### 2.6 Introduction figures/tables

- Optional teaser figure:
  - source-only false positives vs promoted `B1` vs few-shot prediction
- Optional single summary table:
  - source-only, `B1`, `B2 fs20`, upper bound

## 3. Related Work

### 3.1 UAV pavement crack inspection

- Summarize UAV-based pavement crack detection / segmentation work.
- Key angle:
  - many studies optimize in-domain UAV performance
  - fewer explicitly study reuse of ground-trained models under UAV deployment shift

### 3.2 Crack segmentation under domain shift

- Cover cross-dataset or cross-view generalization in crack segmentation and infrastructure imagery.
- Emphasize the gap between benchmark accuracy and deployment transferability.

### 3.3 Hard-negative mining and false-positive suppression

- Review hard-example mining, false-positive reduction, and target-style negative sampling in segmentation/detection.
- Position `B1` as a simple practical adaptation path rather than a full adaptation framework.

### 3.4 Few-shot target adaptation

- Review low-label adaptation / fine-tuning in segmentation.
- Transition into this paper's question:
  how much UAV supervision is actually needed once a realistic deployment gap is established?

### 3.5 Claim boundary sentence

This paper intersects these threads by focusing on deployment diagnosis, structured false positives, and low-cost recovery rather than proposing a new segmentation backbone.

## 4. Problem Setting

### 4.1 Domains and notation

- Source domain: `Crack500`
- Target domain: `UAV_Crack_Segmentation_Kaggle`
- Task: binary crack segmentation
- Label spaces align, but imaging conditions differ substantially

### 4.2 Evaluation settings

Define four reporting settings:

1. Source-only:
train on source domain only, evaluate on target domain

2. `B1` `GT`-filtered target-background / hard-negative mitigation:
augment source training with mined target-domain background / hard-negative crops, without directly optimizing on target-domain positive crack pixels

3. `B2` few-shot:
fine-tune using small labeled target subsets (`fs05 / fs10 / fs20`)

4. Target upper bound:
train and evaluate in-domain on UAV split

### 4.3 Official splits and reporting rule

- Use fixed `UAV train / val / test`
- Use `UAV val` for model selection / threshold selection
- Use `UAV test` only for confirmatory hold-out reporting
- Raw predictions are the official main-table rows
- Postprocessed deployment variants remain comparison-only

### 4.4 Metrics

- `mIoU`
- `F1`
- `precision`
- `recall`

### 4.5 Backbone roles

- `SegFormer-B2`: main headline model
- `DeepLabV3+`: backbone-generality support
- `U-Net`: classical reference / appendix support

## 5. Method

### 5.1 Source-only baselines

- Train source-domain models on `Crack500`
- Evaluate directly on UAV imagery
- Motivation:
  quantify the deployment gap before any mitigation

### 5.2 Temperature-scaled `GT`-filtered target-background / hard-negative mitigation (`B1`)

Core steps:

1. Run source-trained model on target training images
2. Calibrate prediction confidence on `UAV val`
3. Extract connected components from high-confidence target-domain false positives
4. Filter mined crops to avoid overlap with target `GT` crack pixels
5. Mix exported hard-negative crops into source-domain training

Implementation notes to mention:

- target-style crops come from UAV background-only regions
- the method uses target `GT` only to exclude positive crack pixels from the mined crops
- the method does not directly optimize on target-domain positive crack pixels
- bank candidates differ by mining threshold / score configuration
- promoted official `B1` rows:
  - `SegFormer-B2`: `TS-bank thr080_mean082 @ 0.6`
  - `DeepLabV3+`: `TS-bank area1200 @ 0.5`

### 5.3 False-positive audit and nuisance taxonomy

Explain this as an analysis tool, not a new training algorithm.

Two-layer review:

1. Layer 1 bank-quality labels:
`hard_fp`, `ambiguous`, `noise`, `bad_crop`, `gt_issue`

2. Layer 2 nuisance taxonomy:
`pavement_edge`, `shadow_dark_stripe`, `line_like_texture`, `surface_boundary`, `debris_object`, `other_target_clutter`

Paper role:

- diagnose what the model is confusing with cracks
- assess whether mined negatives are meaningful or noisy
- explain why some mined-bank variants help and others do not

### 5.4 Matched random-background control

Purpose:

- test whether `B1` gain comes only from more target-domain background exposure
- compare mined hard negatives against matched random background crops from target images

Required claim boundary in writing:

- control crops are restricted to `GT` background-only regions
- final interpretation is model-dependent, not universally pro-mined

### 5.5 Few-shot target fine-tuning (`B2`)

- Initialize from source-domain checkpoint
- Fine-tune on labeled UAV subsets:
  - `fs05`
  - `fs10`
  - `fs20`
- Use this to quantify supervision efficiency and remaining gap to upper bound

### 5.6 Operating-point selection

- Select thresholds on `UAV val`
- Report confirmatory `UAV test` once per promoted line
- Keep postprocessing separate from the main raw-result table

## 6. Experiments

### 6.1 Setup

- Datasets
- Splits
- Models
- Training details
- Metrics
- Selection rules

### 6.2 RQ1: Zero-shot deployment gap

Main result to show:

- source-only target transfer is poor
- target upper bound is much higher

Core figure/table:

- source-only vs upper bound

### 6.3 RQ2: False-positive diagnosis

Main message:

- source-only failures are structured
- false positives cluster around interpretable nuisance patterns

Core assets:

- audit taxonomy examples
- qualitative case figure

### 6.4 RQ3: Zero-label mitigation with `B1`

Main result:

- `B1` meaningfully improves source-only deployment
- write this carefully as `GT`-filtered target-background / hard-negative mitigation, not as a fully label-free method
- strongest official rows:
  - `SegFormer-B2`: `0.1442 -> 0.3775`
  - `DeepLabV3+`: `0.1230 -> 0.3860`

### 6.5 Mechanism and bank-quality analysis

This subsection should contain both:

1. matched random background vs mined bank
2. audit-driven bank-quality interpretation

Required wording:

- `SegFormer-B2` random background is competitive with or stronger than mined bank on paired reruns
- `SegFormer-B2` curated reruns are diagnostically useful but do not beat the raw mined winner
- `DeepLabV3+` shows positive but cautious evidence for mined-bank-specific value

### 6.6 RQ4: Few-shot recovery

Main result:

- few-shot curve is monotonic
- `SegFormer-B2 fs20` reaches `0.5686`
- upper bound is `0.5879`
- most of the remaining gap is recoverable with modest labels

### 6.7 Backbone generality

- compare headline trend across `SegFormer-B2` and `DeepLabV3+`
- keep `U-Net` compact and secondary

## 7. Discussion

### 7.1 Deployment implications

- source-only reuse is unsafe
- false positives are structured and diagnosable
- low-cost mitigation is helpful even before target annotation
- a staged deployment workflow can guide when to stop at source-only, when to apply `B1`, and when to invest in target annotation

### 7.2 What `B1` does and does not prove

- `B1` clearly helps as a practical mitigation
- bank-quality matters
- mined-bank-specific mechanism is not universal

### 7.3 Why few-shot matters

- small target supervision is highly effective
- useful for practical annotation budgeting

### 7.4 Limitations

- single target dataset
- no formal domain adaptation baseline family
- audit is partly manual
- some mechanism evidence is model-dependent

## 8. Conclusion

### 8.1 Closing claim

Ground-to-UAV crack segmentation should be studied as a deployment problem with diagnosable failure modes and practical recovery paths.

### 8.2 Final takeaways

1. Source-only cross-domain performance is not deployment-ready.
2. False positives are structured rather than random.
3. `GT`-filtered target-background / hard-negative mitigation provides a practical low-cost improvement path without directly optimizing on target-domain positive crack pixels.
4. Limited UAV annotation closes most of the remaining gap.

## 9. Figure And Table Checklist

### Main paper

1. Main comparison table:
source-only, `B1`, `B2`, upper bound

2. Supervision-scaling figure:
already available in `results/report_assets/final_comparison/supervision_scaling.png`

3. Threshold / `B1` hold-out figure:
already available in `results/report_assets/b1_holdout/threshold_sweep_val.png`

4. Qualitative progression figure:
already available in `results/report_assets/final_comparison/qualitative_progression_test.png`

5. Deployment workflow figure:
- source-only evaluation -> false-positive diagnosis -> `B1` mitigation -> few-shot escalation if needed

6. SegFormer audit summary figure:
use `notebooks/10_segformer_audit_followup.ipynb`

### Appendix or supplement

1. Full audit label distributions
2. Extra review-card examples
3. Historical comparison-only operating points
4. Optional U-Net details

## 10. Citation Slots To Fill While Writing

Do not invent citations during drafting. Fill these slots with the papers you already collected:

1. UAV pavement crack inspection / UAV-based pavement condition assessment
2. Pavement crack segmentation benchmark papers
3. Cross-domain or cross-dataset crack detection / segmentation
4. Hard-negative mining / false-positive suppression in dense prediction
5. Few-shot or low-label segmentation adaptation
6. Calibration / confidence estimation in segmentation
