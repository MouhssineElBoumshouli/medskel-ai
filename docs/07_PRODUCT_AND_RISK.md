# 07 — Long-term product, users, regulatory and risk boundaries

## Product hypothesis (NOT validated)

A retinal vessel analysis research tool may be valuable if it helps ophthalmology/medical imaging researchers obtain **consistent, transparent, inspectable measurements** from sets of fundus photographs. The proposed positioning is 'research-use vascular morphometry and reproducibility', not 'AI diagnoses disease'.

Potential flow: batch import permitted/de-identified images → quality checks → segment vessels → inspect/correct mask → trace centerlines → measure agreed endpoints → export clean CSV/JSON + overlays + metadata/warnings. Every step should state model version, preprocessing, image resolution and known uncertainty.

### Candidate researchers and experiments to understand their real needs

- Retinal image segmentation lab: how do they compare predicted masks and annotations today? Where do analyses break?
- Epidemiology/clinical cohort analyst: which standardized measurements, regions, units and reproducibility standards do they actually need? What formats and sample throughput?
- Ophthalmology research staff: how often are poor images excluded, manually corrected or adjudicated? What evidence would persuade them that a metric is useful?
- Computational imaging research group: how important are explainability, reproducibility, deployment limitations and batch workflows compared with existing toolchains?

Ask about actual past work and concrete failure cases; avoid leading questions like 'Would you pay for AI?'. Get permissions for any interview or outside messages first.

## Competing offerings are substantial

AutoMorph and VascX already automate segmentation, quality control and vessel biomarkers. A generic 'retina photo → vessel measurement dashboard' is not a defensible advantage. Potential differentiation **hypotheses** to test:

1. More trustworthy *repeatable measurements* on difficult images without eliminating fine vessels.
2. User-auditable mask/graph corrections and transparent per-image provenance with lower manual review time.
3. Measurable cross-camera robustness/quality control and known abstention conditions.
4. Reproducible research pipelines that are easier to integrate than existing tools, possibly with openly specified outputs.

If controlled comparisons find no measurable improvement or customer interest, narrow/reconsider the product rather than forcing a startup story.

## Basic validation stages

1. **Technical:** reproducible CNN segmentation and robust geometry; tests, held-out errors and baselines.
2. **Scientific:** independent annotations, fair paired comparisons, multiple quality strata, camera effects, group-aware CIs and transparent failure analysis.
3. **Workflow:** researcher interviews, current process mapping, test batch import/export, time-to-fix and study traceability.
4. **Economics:** cost of compute/storage/review, realistic willingness to pay, licensing, support burden and procurement requirements.
5. **Legal and ethics:** evaluate intended use and jurisdictions with qualified advice; privacy, licensing, research consent and relevant approval needs. Do not infer regulatory classification solely from word 'research'.

## Regulatory/medical guardrails

- Initial system is **research software**, not a medical device or instrument. This label alone doesn't exempt later deployments from regulation; classification follows intended use and jurisdiction.
- Do not claim diabetes diagnosis, blood-pressure estimation, risk prediction, treatment recommendation or clinical decision support without dedicated studies and regulatory/legal review.
- Risk-factor prediction literature is not sufficient to turn a vessel segmentation project into a clinical predictor.
- Identify health data and remove names/metadata when appropriate. No identifiable patient image or record uploaded to external AI, cloud, public repo or third-party service without informed permission, legal basis, ethical approval and a data protection plan.
- Do not market a pixel width as micrometres. If physical calibration is unavailable, report pixel units with warnings.
- Investigate GDPR, Morocco's personal data law and health-data rules, target countries and institutional research governance **when intended jurisdiction and processing are known**; do not assert compliance from a generic checklist.
- Current VascX software is listed under PolyForm Noncommercial; training masks and model weights may be under distinct rights. Do a separate license inventory before any integration or commercial distribution.

## Risks, severity and mitigation

| Risk | Why important | Immediate action |
|---|---|---|
| No genuine trained CNN yet | No usable class demo | Prioritize real FIVES pair loader and tiny-batch training, then valid holdout |
| CPU-only environment | Large high-resolution training may be slow | Benchmark small patches/size and reduce model carefully; do not hide thin-vessel tradeoffs |
| Unknown FIVES patient mapping | Repeated subjects may leak | Check metadata, SHA, IDs; disclose limits; consider stricter subject-identified external set |
| Thin vessels vanish on 128px resize | Attractive mask but bad anatomical coverage | Inspect native vs resized masks; train patches/higher resolution when feasible |
| MedSkel simplification hides details | Reproducible but wrong measurements | Track absolute bias, fine-vessel recall, coverage and centerline position |
| Numerical geometry failures | Invalid graph/width at crossings | Unit-test topology, disconnected components, coordinates, empty masks, runtime |
| Misleading disease/BP claims | Scientific, ethical and regulatory error | Confine claim to segmentation/measurements; cite study distinctions |
| Competition/novelty weak | Low commercial value | Benchmark AutoMorph/VascX and interview users before product build |
| License conflict | Ineligible redistribution/commercial model | Verify software/model/dataset licenses per exact version |
| Clinical dataset/privacy issue | Data breach or unauthorized use | Public de-identified datasets only initially; approvals for any other data |
| Evaluation leakage | Invalid performance claims | Immutable split manifest, train-only tuning, locked test set |
| Student can't explain generated code | Fails CNN course goal | Teach math and require a demo and code review in plain language |

## Honest business hypothesis statement

'If researchers need reliable retinal vessel measurements and current tools fail on a specific repeatability or review workflow, MedSkel AI may be a useful research tool. We will validate this through controlled technical tests and interviews. We currently have **no verified paying customers, clinical validation, or demonstrated competitive advantage**.'
