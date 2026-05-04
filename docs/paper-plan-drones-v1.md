# Drones-Facing Plan v1

**Last updated:** 2026-05-04  
**Positioning:** `Drones`-oriented fallback / parallel writing plan  
**Primary evidence base:** [paper-plan-v4.md](./paper-plan-v4.md), [research-plan.md](./research-plan.md), [baseline-results.md](./baseline-results.md), [results/report_assets/final_comparison/summary.md](../results/report_assets/final_comparison/summary.md)

## 1. Core Positioning

For `Drones`, the paper should be framed less as an infrastructure deployment workflow paper and more as a **UAV-domain transfer and low-cost recovery study for pavement crack segmentation**.

The same frozen experiments remain valid, but the narrative emphasis changes:

- foreground UAV image acquisition differences such as viewpoint, scale, altitude, texture compression, and clutter
- treat source-to-target failure as a **ground-to-UAV visual domain shift**
- present `B1` as a practical **`GT`-filtered target-background / hard-negative mitigation** step
- present `B2` as an **annotation-efficiency study** for UAV adaptation
- keep false-positive audit as UAV failure diagnosis rather than as construction workflow guidance

Working one-sentence framing:

> This paper studies how pavement crack segmentation models trained on ground-view imagery transfer to UAV imagery, why structured false positives emerge under this cross-view domain shift, and how much performance can be recovered through low-cost `GT`-filtered target-background / hard-negative mitigation and limited UAV supervision.

## 2. Recommended Drones Angle

The strongest `Drones` version is not:

- a new segmentation architecture paper
- a pure hard-negative mining method paper
- a generic deployment workflow paper

The strongest `Drones` version is:

- a controlled **ground-to-UAV transfer study**
- with **UAV failure-mode diagnosis**
- plus **low-cost adaptation and annotation-budget evidence**

In other words, the paper should answer:

1. How badly do ground-trained crack segmentation models fail on UAV imagery?
2. What kinds of UAV nuisance structures trigger high-confidence false positives?
3. Can background-only target exposure help before dense UAV annotation is available?
4. How little UAV annotation is needed to recover most of the remaining gap?

## 3. Drones-Specific Claim Set

This paper **does claim**:

- UAV imagery introduces a severe cross-view / cross-scale transfer gap for ground-trained crack segmenters
- the transfer failure is structured and diagnosable rather than random
- `GT`-filtered target-background / hard-negative mitigation can substantially improve UAV transfer performance without directly optimizing on target-domain positive crack pixels
- few-shot UAV supervision closes most of the remaining gap efficiently
- the transfer story is visible across both Transformer and CNN backbone families

This paper **should state carefully**:

- `B1` is not a fully label-free method
- the method uses target `GT` only to exclude positive crack pixels from mined target crops
- mined-bank-specific value is model-dependent rather than universal
- `SegFormer-B2` random-background controls weaken any overly strong "mined negatives are uniquely better" claim

This paper **does not need to claim**:

- a new state-of-the-art crack segmentation architecture
- a universal theory of hard-negative mining
- a fully automated UAV adaptation pipeline
- field-validated deployment at production scale

## 4. Recommended Title Directions

Most `Drones`-compatible titles:

1. Ground-to-UAV Pavement Crack Segmentation Under Domain Shift: Hard-Negative Mitigation and Few-Shot Recovery
2. Diagnosing Ground-to-UAV Transfer Failure for UAV Pavement Crack Segmentation
3. Cross-View Domain Shift in UAV Pavement Crack Segmentation: Failure Diagnosis, Low-Cost Mitigation, and Few-Shot Adaptation
4. From Ground Imagery to UAV Crack Segmentation: Structured False Positives and Low-Label Recovery

If the journal version wants a cleaner UAV emphasis, the safest default is:

> Cross-View Domain Shift in UAV Pavement Crack Segmentation: Failure Diagnosis, Low-Cost Mitigation, and Few-Shot Adaptation

## 5. Abstract Strategy

The abstract should lean more on UAV imaging conditions than the `AIC` version.

Suggested five-part structure:

1. Motivation:
UAV imagery is attractive for pavement inspection, but target-domain UAV labels are expensive and many crack models are trained on ground-view datasets.

2. Gap:
It remains unclear how well ground-trained segmenters transfer to UAV imagery and what kinds of UAV-specific false positives dominate this failure.

3. Study design:
Benchmark source-only transfer from `Crack500` to `UAV_Crack_Segmentation_Kaggle`, analyze false positives with audit and taxonomy, evaluate `GT`-filtered target-background / hard-negative mitigation (`B1`), and measure few-shot recovery (`B2`).

4. Key numbers:

- `SegFormer-B2` source-only `IoU 0.1442`
- promoted `SegFormer-B2 B1` `IoU 0.3775`
- promoted `DeepLabV3+ B1` `IoU 0.3860`
- `SegFormer-B2 fs20` `IoU 0.5686`
- target upper bound `IoU 0.5879`

5. Interpretation:
Ground-to-UAV transfer failure is severe but largely recoverable; structured false positives are a major UAV-side failure mode; low-cost negative exposure helps, and modest UAV annotation recovers most of the remaining gap.

## 6. Section Weighting For Drones

Compared with `AIC`, the `Drones` version should reweight the paper as follows.

### More central in Drones

- UAV acquisition differences
- cross-view / cross-scale transfer failure
- UAV nuisance structures and qualitative failure cases
- supervision efficiency for UAV adaptation
- robustness-oriented discussion

### Less central in Drones

- deployment workflow rhetoric
- construction-management decision language
- annotation-budget guidance as a managerial contribution

### Still important, but reframed

- false-positive audit:
  write this as UAV failure analysis
- random background vs mined bank:
  keep as mechanism boundary, not headline
- `B1`:
  write as low-cost target-background / hard-negative exposure for UAV transfer

## 7. Frozen Result Backbone

The `Drones` paper should use the same frozen result backbone as the `AIC` plan.

### Main headline model

- `SegFormer-B2`
  - source-only raw: `IoU 0.1442`
  - promoted `B1`: `TS-bank thr080_mean082 @ 0.6`, `IoU 0.3775`
  - `B2`: `fs05 0.5074`, `fs10 0.5420`, `fs20 0.5686`
  - target upper bound: `0.5879`

### Generality support

- `DeepLabV3+`
  - source-only raw: `IoU 0.1230`
  - promoted `B1`: `TS-bank area1200 @ 0.5`, `IoU 0.3860`
  - `B2`: `fs05 0.3599`, `fs10 0.4354`, `fs20 0.4760`

### Classical reference

- `U-Net`
  - source-only raw: `IoU 0.1284`

## 8. How To Write B1 In The Drones Version

Use careful language throughout:

- preferred:
  - `GT`-filtered target-background / hard-negative mitigation
  - no-target-positive `B1`
  - background-only target exposure

- avoid as the default wording:
  - fully label-free adaptation
  - zero-label target adaptation

Recommended sentence:

> `B1` is a low-cost mitigation step that uses `GT`-filtered target-domain background / hard-negative crops without directly optimizing on target-domain positive crack pixels.

## 9. How To Write The Mechanism Section

The safest `Drones` interpretation is:

- structured false positives matter
- extra target-background exposure helps
- mined-bank quality still matters
- the specific advantage of mined over matched random background is backbone-dependent

Use the evidence this way:

### SegFormer-B2

- paired `5-seed` random-background controls are competitive with or slightly stronger than mined-bank reruns on `UAV val`
- therefore do not write `SegFormer-B2` as proof that mined negatives are uniquely superior

### DeepLabV3+

- the single matched control remains below the mined validation winner
- therefore `DeepLabV3+` still gives a directional hint that mined-bank quality can matter
- state this cautiously because the control quality is weaker than the `SegFormer-B2` paired comparison

Recommended phrasing:

> The mechanism evidence is claim-bounded rather than one-sided: generic target-background exposure explains a substantial share of the gain on `SegFormer-B2`, while `DeepLabV3+` still suggests that mined-bank quality can matter when visually meaningful nuisance structures are retained.

## 10. Role Of The Audit In Drones

In the `Drones` version, false-positive audit should support a UAV-imaging interpretation:

- pavement edges
- line-like texture
- shadow / dark stripe
- debris or clutter
- surface boundary effects

The audit is valuable because it shows that UAV transfer failure is not only a metric drop, but also a **qualitatively structured misinterpretation of UAV surface patterns**.

The `SegFormer-B2` audit result should be written honestly:

- curated banks improve interpretability
- removing `ambiguous` helps on `thr080_mean082`
- but curated reruns do not beat the raw mined validation winner

So the audit contributes:

- explanation
- taxonomy
- bank-quality evidence

not:

- a new stronger training recipe

## 11. Figure And Table Priorities

For `Drones`, the paper should prioritize visual and UAV-facing assets.

### Must-have figures

1. Main results figure or table:
source-only -> `B1` -> `B2` -> upper bound

2. Few-shot efficiency curve:
`SegFormer-B2` and `DeepLabV3+` versus labeled UAV samples

3. Qualitative UAV progression figure:
source-only vs promoted `B1` vs `B2` on representative UAV cases

4. False-positive audit figure:
representative `hard_fp`, `ambiguous`, and `noise` UAV crops

5. Optional mechanism figure:
matched random background vs mined-bank comparison as a claim-boundary figure

### Must-have tables

1. Main comparison table
2. Audit taxonomy summary table
3. Curated-bank follow-up summary table for `SegFormer-B2`

The notebook [10_segformer_audit_followup.ipynb](../notebooks/10_segformer_audit_followup.ipynb) should be treated as a figure-prep asset for this version.

## 12. Recommended Section Flow

The `Drones` manuscript should flow like this:

1. Introduction:
UAV inspection motivation, reuse temptation, and cross-view risk

2. Related Work:
UAV crack inspection, crack segmentation transfer, hard-negative mitigation, low-label UAV adaptation

3. Problem Setting:
source domain, UAV target domain, official splits, four reporting settings

4. Method:
source-only baseline, `B1`, audit, random-background control, `B2`

5. Experiments:
setup and reporting rule

6. Results:
zero-shot transfer gap, false-positive diagnosis, `B1`, mechanism boundary, few-shot recovery

7. Discussion:
what this says about UAV transferability, nuisance structure, and low-label recovery

8. Conclusion

## 13. If There Is Time For Light Drones-Strengthening

No large new experiment family is strictly required, but these additions would fit `Drones` especially well if already easy to prepare from existing assets:

1. a compact UAV acquisition discussion:
viewpoint, altitude, apparent crack width, and clutter differences

2. a small robustness appendix:
resolution or contrast stress examples if already available

3. a stronger qualitative UAV figure set:
showing crack fragmentation, false positives, and recovery

These are strengthening items, not blockers.

## 14. Final Recommendation

Use the `AIC` version when the main selling point is:

- deployment reliability
- diagnostic workflow
- practical mitigation guidance

Use the `Drones` version when the main selling point is:

- UAV visual domain shift
- structured cross-view transfer failure
- low-cost UAV adaptation
- few-shot recovery efficiency

The two versions should share the same frozen experiments, but differ in emphasis rather than in claimed evidence.
