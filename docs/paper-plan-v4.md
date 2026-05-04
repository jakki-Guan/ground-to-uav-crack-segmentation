# Paper-Facing Plan v4

**Last updated:** 2026-05-04  
**Submission strategy:** `Automation in Construction` first, `Drones` second

This document is the paper-facing companion to [research-plan.md](./research-plan.md).  
It does **not** replace the long-form experiment archive. Instead:

- [research-plan.md](./research-plan.md) remains the long-term research reference and experiment log
- [baseline-results.md](./baseline-results.md) remains the curated frozen-result record
- this file defines the paper positioning, claim boundaries, priority experiments, and writing roadmap

## 1. Updated Positioning

This paper is positioned as a controlled diagnostic study of **ground-to-UAV pixel-level crack segmentation**.

Rather than proposing a new segmentation architecture, it:

- quantifies the zero-shot domain gap from ground imagery to UAV imagery
- diagnoses the dominant false-positive failure modes behind that gap
- evaluates a `GT`-filtered target-background / hard-negative mitigation strategy (`B1`)
- measures how efficiently few-shot target supervision (`B2`) closes the remaining gap

Working one-sentence framing:

> Ground-trained crack segmentation models can fail when deployed in UAV-assisted pavement inspection workflows. This study quantifies the deployment gap, diagnoses high-confidence false positives, evaluates low-cost `GT`-filtered target-background / hard-negative mitigation, and measures how rapidly limited UAV annotation recovers performance.

## 2. What Stays From the Existing Plan

The existing project logic remains valid and should be treated as the experimental backbone:

- controlled benchmark / evaluation study rather than a new-model paper
- `Crack500` as the source-domain training dataset
- `UAV_Crack_Segmentation_Kaggle` as the target-domain evaluation dataset
- `SegFormer-B2`, `U-Net`, and `DeepLabV3+` as representative baselines
- raw predictions as the official baseline rows
- postprocessed variants reported separately as deployment-oriented operating points
- one-change-at-a-time ablation logic where relevant
- `mIoU`, `F1`, `precision`, and `recall` as the main metrics

## 3. What Changes in the Paper Narrative

The paper should no longer be led primarily by cross-resolution or illumination robustness language.

Those topics can remain as:

- appendix robustness probes
- UAV stress-test extensions
- Drones-oriented strengthening experiments
- future work

The main paper chain should instead be:

1. source-only ground-to-UAV transfer fails
2. failure is driven heavily by target-domain false positives
3. `GT`-filtered target-background / hard-negative mitigation improves that failure mode
4. few-shot target supervision closes most of the remaining gap quickly

## 4. Core Research Questions

### RQ1. Zero-shot deployment gap

How large is the performance drop when a crack segmentation model trained on ground imagery is applied directly to UAV imagery?

### RQ2. Failure diagnosis

Are the dominant source-only failures random, or are they concentrated in interpretable high-confidence false-positive nuisance patterns?

### RQ3. Low-cost mitigation

Can `GT`-filtered mined hard negatives from the target domain improve deployment performance without requiring dense target-domain retraining?

### RQ4. Supervision efficiency

How much target-domain annotation is needed to recover most of the remaining performance gap after source-only or `B1` transfer?

## 5. Claim Boundaries

This paper **does claim**:

- source-only ground-to-UAV crack segmentation is not deployment-ready
- false-positive nuisance patterns are a major and diagnosable source of failure
- `GT`-filtered target-background / hard-negative crops can provide real mitigation without directly optimizing on target-domain positive crack pixels
- a small amount of UAV supervision closes the remaining domain gap very quickly
- the main transfer story is visible across more than one backbone family

This paper **does not claim**:

- a new state-of-the-art segmentation architecture
- a full domain adaptation method with theoretical guarantees
- that postprocessing is the main solution after `B1`
- that every temperature-scaled mining variant is helpful or that calibrated-bank selection is solved independently of bank quality

## 6. Existing Evidence Already Strong Enough for the Paper

These result families should now be treated as the main frozen evidence base:

### SegFormer-B2 line

- source-only raw cross-domain baseline
- promoted `TS-bank B1` hold-out row at `0.6`
- historical raw `B1` calibration rows at `0.5 / 0.7 / 0.9` as comparison context
- target-domain upper bound
- `B2` few-shot curve: `fs05 / fs10 / fs20`

### DeepLabV3+ line

- source-only raw cross-domain baseline
- promoted `TS-bank area1200 B1` hold-out row at `0.5`
- `B2` few-shot curve: `fs05_pat12 / fs10 / fs20`

### Classical reference

- `U-Net` as the stabilized CNN reference rather than the main paper storyline

### Reporting rule

- postprocessed rows remain comparison-only deployment variants

## 7. Role of Each Backbone in the Paper

### SegFormer-B2

Main backbone for the headline diagnostic and mitigation story.

### DeepLabV3+

Backbone-generality support:

- shows the domain-gap and mitigation pattern is not unique to one architecture family
- strengthens `B1/B2` claims beyond the Transformer line

### U-Net

Classical reference:

- useful for repository completeness, appendix support, and historical baseline context
- should not dominate the main paper story unless a reviewer explicitly asks for it

## 8. AIC-First Framing

If targeting `Automation in Construction`, the paper should foreground deployment reliability in infrastructure inspection.

Preferred framing:

- UAV-assisted pavement inspection is valuable in practice
- source-trained models may fail under deployment domain shift
- those failures should be diagnosed rather than only benchmarked
- low-cost adaptation paths matter because dense target labels are expensive

The `AIC` lens cares about:

- why deployment fails
- whether the failure is interpretable
- whether mitigation is cheap and practical
- whether a small amount of UAV annotation is enough
- whether the paper offers workflow guidance for infrastructure inspection

## 9. Drones-Second Framing

If `AIC` is not the final venue and the paper is redirected toward `Drones`, UAV acquisition differences should become more visible in the framing.

Preferred emphasis:

- UAV image acquisition conditions
- flight height and viewpoint effects
- target-domain nuisance structures
- resolution and lighting shifts
- reliability of fully automatic transfer in drone inspection workflows

In that version, robustness stress tests become more central rather than secondary.

## 10. Priority Experiment Groups

### Group A: Freeze Official Results

These should be treated as the frozen official core rows:

1. `SegFormer-B2` source-only raw
2. promoted `SegFormer-B2 TS-bank B1` at `0.6`, with raw `0.5 / 0.7 / 0.9` kept as comparison context
3. `SegFormer-B2` UAV upper bound
4. `SegFormer-B2 B2` at `fs05 / fs10 / fs20`
5. `DeepLabV3+` source-only raw
6. promoted `DeepLabV3+ TS-bank area1200 B1` at `0.5`
7. `DeepLabV3+ B2` at `fs05_pat12 / fs10 / fs20`
8. `U-Net` as a classical reference
9. all postprocess rows as comparison-only

### Group B: AIC-Strengthening Experiments

Highest-value remaining additions for the paper:

- Completed mechanism check: matched random background vs mined hard-negative bank now narrows the `B1` claim boundary rather than cleanly proving a mined-signal advantage.
1. false-positive audit and nuisance taxonomy
2. threshold sensitivity and bank-quality analysis
3. optional cross-bank transfer

### Group C: Drones-Strengthening Experiments

Good fallback or extension items:

1. resolution stress test
2. illumination / contrast stress test
3. UAV acquisition discussion
4. qualitative UAV deployment examples

## 11. Immediate Missing Pieces

### Completed Study. Random background bank vs mined bank

Purpose:

- show that `B1` is not simply "more UAV background"
- test whether mined high-confidence false positives carry specific value

Current result:

- `SegFormer-B2`: the initial single random-background control was much lower than the mined bank (`0.2747 < 0.3693`), but paired `5-seed` reruns reverse the average:
  - matched random-background mean `IoU 0.3401 +- 0.0215`, `F1 0.4791 +- 0.0237`
  - mined `TS-bank` mean `IoU 0.3194 +- 0.0239`, `F1 0.4584 +- 0.0247`
- `DeepLabV3+`: a single matched random-background control reaches `IoU 0.3023`, `F1 0.4368`, below the mined `component_area >= 1200` validation row at `IoU 0.3564`, `F1 0.5136`
- all random control crops were explicitly restricted to `GT` background-only regions, so this control does not leak positive crack supervision into the comparison

Interpretation:

- `SegFormer-B2` no longer supports a strong "mined negatives are uniquely better" mechanism claim; generic target-background exposure already explains a substantial share of the gain on this backbone
- `DeepLabV3+` is directionally consistent with useful mined nuisance signal, but that evidence should still be stated cautiously because its larger crop sizes forced cross-image fallback on `25 / 28` control crops
- the safest paper-facing reading is therefore model-dependent rather than universal: `random background` is a strong control and can match or exceed mined-bank transfer on `SegFormer-B2`, while `DeepLabV3+` still shows a weaker but positive mined-bank hint

### Priority 1. False-positive audit / nuisance taxonomy

Purpose:

- explain what the model is actually confusing with cracks
- turn failure modes into engineering-readable categories

Current status:

- a first `SegFormer-B2` audit package now exists at `results/hard_negative_audit/segformer_b2_tsbank_round1`
- the package covers `thr080_mean082`, `thr080_mean080`, and `thr075_mean080` with `180` reviewed cards (`60` per bank)
- reviewed keepable shares (`hard_fp + ambiguous`) are:
  - `thr080_mean080`: `20 / 60 = 33.3%`
  - `thr080_mean082`: `17 / 60 = 28.3%`
  - `thr075_mean080`: `12 / 60 = 20.0%`
- audit-derived curated-bank reruns on `SegFormer-B2` are:
  - `thr080_mean080 hard_fp + ambiguous`: `IoU 0.3439`, `F1 0.4830`
  - `thr080_mean080 hard_fp`: `IoU 0.3420`, `F1 0.4836`
  - `thr080_mean082 hard_fp + ambiguous`: `IoU 0.3304`, `F1 0.4682`
  - `thr080_mean082 hard_fp`: `IoU 0.3535`, `F1 0.4993`
- interpretation:
  - the strongest curated export is `thr080_mean082 hard_fp`, but it still remains below the raw mined validation winner `segformer_b2_b1_tsbank_thr080_mean082` at `IoU 0.3693`
  - removing `ambiguous` crops helps materially on `thr080_mean082`, but not on `thr080_mean080`
  - the present value of the `SegFormer-B2` audit is therefore explanatory rather than promotable: it sharpens the bank-quality story, but it does not replace the current promoted `B1` row

Suggested categories:

- pavement edge
- shadow / dark stripe
- line-like texture
- surface boundary
- debris / object
- bad crop
- `GT` issue

### Priority 2. Bank-quality analysis

Purpose:

- connect `TS-bank` instability to mined sample quality
- verify whether selected negatives are meaningful high-confidence false positives or mostly noise

Current status:

- a first `DeepLabV3+ TS-bank` audit package already exists at `results/hard_negative_audit/deeplabv3plus_tsbank_round1`
- the audit package has already been upgraded to a two-layer review schema:
  - layer 1 = bank-quality judgment
  - layer 2 = nuisance taxonomy for paper-facing failure diagnosis
- the manual review was completed on `2026-05-02` with `223 / 223` layer-1 labels and `222 / 223` layer-2 labels filled
- aggregate layer-1 review distribution is `noise = 129`, `hard_fp = 66`, `ambiguous = 27`, `bad_crop = 1`
- reviewed keepable share (`hard_fp + ambiguous`) is highest for the calibrated-bank baseline at `23 / 43 = 53.5%`, but all audited `DeepLabV3+` banks still include substantial noise
- `export_curated_hard_negative_bank.py` now converts the reviewed `audit_samples.csv` rows into trainable curated bank roots
- curated-bank validation follow-up shows:
  - calibrated `TS-bank` filtered to `hard_fp + ambiguous`: `IoU 0.2920`, `F1 0.4301`
  - calibrated `TS-bank` filtered to `hard_fp` only: `IoU 0.3099`, `F1 0.4567`
  - threshold sweep on the curated `hard_fp` checkpoint peaks at `IoU 0.3122` with threshold `0.40`
- `export_auto_filtered_hard_negative_bank.py` now exports simple audit-derived rule variants of the same calibrated bank
- auto-filter validation follow-up shows:
  - `span_ratio >= 0.35`: keeps `23 / 43` audited crops (`15 hard_fp`, `3 ambiguous`, `5 noise`) and reaches `IoU 0.2783`, `F1 0.4086`
  - `component_area >= 1200`: keeps `28 / 43` audited crops (`16 hard_fp`, `4 ambiguous`, `8 noise`) and reaches `IoU 0.3564`, `F1 0.5136`
- interpretation:
  - removing `ambiguous` crops helps, so the ambiguous subset is not harmless
  - `span_ratio` alone is too aggressive and underperforms despite its intuitive semantics
  - `component_area >= 1200` is the first simple audit-derived rule that clearly beats both the raw calibrated bank (`0.3564 > 0.3149`) and the best curated operating point (`0.3564 > 0.3122`)
  - the paper-facing story should therefore shift from a generic purity-versus-diversity tradeoff to a more specific observation: removing small noisy components helps, but collapsing the bank to only the visually cleanest `hard_fp` crops removes useful target-domain nuisance diversity
- in contrast, the later `SegFormer-B2` audit-derived curated reruns are useful as diagnostics but not as improvements:
  - `thr080_mean082 hard_fp` is the strongest curated export at `IoU 0.3535`, but it still trails the raw mined validation winner `0.3693`
  - `ambiguous` crops dilute the `thr080_mean082` bank, whereas the `thr080_mean080` curated exports are nearly tied
- `deeplabv3plus_b1_tsbank_autofilter_area1200_test` has now confirmed on the fixed hold-out split at `IoU 0.3860`, `F1 0.5470`, so it should be treated as the official zero-label `DeepLabV3+ B1` row rather than as a pending candidate

### Priority 3. Optional cross-bank transfer

Purpose:

- test whether a model-specific mined bank captures architecture-specific nuisance patterns
- or instead captures more general target-domain clutter

## 12. Paper Outline

### 1. Introduction

- motivate UAV pavement inspection
- state the risk of deploying ground-trained segmentation models directly on UAV imagery
- position the work as a diagnostic and mitigation study rather than a new architecture paper

### 2. Related Work

1. UAV pavement crack segmentation and cross-resolution methods
2. crack segmentation domain adaptation
3. hard-negative mining and cross-domain false positives
4. calibration, confidence, and segmentation uncertainty

### 3. Problem Setting

- source domain: `Crack500`
- target domain: UAV split
- source-only evaluation
- `B1` `GT`-filtered target-background / hard-negative mitigation
- `B2` few-shot fine-tuning
- target-domain upper bound
- official hold-out split

### 4. Method

- source-only baselines
- `B1`: `GT`-filtered target-background / hard-negative bank mixed training
- `B2`: few-shot target-domain fine-tuning
- calibrated operating-point selection and reporting rules

### 5. Experiments

1. in-domain source and target references
2. zero-shot cross-domain degradation
3. false-positive failure diagnosis
4. `B1` mitigation
5. random background vs mined bank
6. `B2` few-shot recovery
7. backbone generality
8. optional resolution stress test

### 6. Discussion

- source-only transfer is not deployment-ready
- false positives are structured rather than random
- `GT`-filtered target-background / hard-negative mitigation helps, but bank-quality effects are model-dependent
- few-shot labels close the gap quickly
- practical implications for `AIC` and `Drones` audiences

## 13. Recommended Repository-Level Document Roles

### File 1: `docs/research-plan.md`

Use as:

- long-term research reference
- experiment log
- implementation and result archive
- place for caveats, reruns, and historical branches

### File 2: `docs/paper-plan-v4.md`

Use as:

- paper-facing positioning document
- submission strategy note
- experiment prioritization guide
- claim-boundary reference while writing

### File 3: `docs/paper-outline-detailed-v1.md`

Use as:

- the detailed `AIC` writing outline
- section-by-section drafting scaffold
- figure / table checklist for the main submission path

### File 4: `docs/paper-draft-intro-problem-method-v1.md`

Use as:

- the first `AIC`-oriented prose draft
- the canonical wording reference for `B1`
- the baseline text to extend into `Related Work`, `Experiments`, and `Results`

### File 5: `docs/paper-plan-drones-v1.md`

Use as:

- a venue-specific fallback / parallel plan for `Drones`
- a reweighting guide when the same frozen experiments are written from the UAV-visual-domain-shift angle
- the main reference for `Drones` title, abstract, and section emphasis

## 14. 2026-05-04 Writing Checkpoint

The experimental core is now sufficiently stable for venue-specific drafting, and the repository has moved from experiment-first mode into paper-production mode.

Completed writing assets:

- `AIC`-oriented detailed outline: `docs/paper-outline-detailed-v1.md`
- `AIC`-oriented `Introduction / Problem Setting / Method` draft: `docs/paper-draft-intro-problem-method-v1.md`
- `Drones`-oriented venue plan: `docs/paper-plan-drones-v1.md`
- compact `SegFormer` audit figure-prep notebook: `notebooks/10_segformer_audit_followup.ipynb`

## 15. Final Summary

The repository should now be organized around a two-layer planning model:

- **experiment archive layer**: detailed, historical, and exhaustive
- **paper-facing layer**: selective, claim-driven, and venue-aware

Working final thesis:

> Ground-to-UAV crack segmentation should be framed as a deployment diagnosis problem: source-only failure, false-positive audit, `GT`-filtered target-background / hard-negative mitigation, and few-shot recovery together provide a practical roadmap for reliable UAV pavement inspection.
