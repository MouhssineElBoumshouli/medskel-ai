# MedSkel AI — Project charter (proposal, not product validation)

## One-line mission
Develop a reproducible retinal fundus image pipeline that produces **auditable vessel segmentations and measurements** for research, with evidence of reliability relative to existing software.

## Current phase
Discovery + experimental design. A medical product is *not* yet established. University CNN course is an opportunity to perform an authentic first study.

## Specific initial customer hypothesis
Researchers processing collections of color retinal photographs need measurements they can trust, inspect, correct and export with less manual effort. This is an assumption, to test with real research workflows.

## Proposed workflow (not all implemented)
1. Obtain a consented or public color fundus image and provenance metadata.
2. Check field of view and image quality; abstain from poor images when necessary.
3. Use CNN to segment vessel pixels into a probabilistic mask.
4. Convert mask to an interpretable vessel graph via MedSkel; compare with pixel-thinning and modern graph tools.
5. Derive only validated measurements, with region definitions and units.
6. Provide image overlays, failure flags and reviewer correction.
7. Export per-image results, provenance, model version and warnings.

## What exists today (in original medskel repo)
- Geometric bisector/Voronoi skeletonization and pixel-thinning baseline, plus metrics/tests.
- Retinal/skull examples; CHASE_DB1 two-observer study.
- Reported pixel-space total vessel length, median estimated calibre, tortuosity, branch count and bifurcation count.
- No trained CNN in original dependencies; no validated standardized CRAE/CRVE/AVR.

## First product-risk hypotheses to challenge
- Hypothesis A: A learned segmentation model improves *downstream vessel measurement accuracy* over hand-engineered masks.
- Hypothesis B: MedSkel provides a meaningful advantage beyond classical thinning and modern research tools in a specific measurement regime.
- Hypothesis C: Researchers have an unmet audit/review/throughput problem and would pilot/pay for a solution.
- Hypothesis D: Product can run under reasonable compute and privacy constraints.

## Non-goals now
- Diagnosing any disease; risk scores; handling real identified patient records; operating a camera/scanner; cloud medical service; clinical regulatory claims; designing an elaborate dashboard before solving the measurement problem.

## Definitions requiring care
- Segmentation Dice measures agreement of vessel pixels, not correctness of width/bifurcation measurements.
- CHASE observer agreement measures expert disagreement; neither observer is absolute clinical truth.
- Inter-observer agreement is different from repeat imaging on another device.
- Pixel measurements are not micrometers without calibration.
- CRAE/CRVE/AVR require artery/vein identification and standardized optic-disc-centered protocols.

## Evidence gates before commercialization
1. External dataset performance with errors and confidence intervals (not cherry-picked images).
2. Validated biomarker or defined measurements versus independent reference/annotation.
3. Comparative analysis against AutoMorph and VascX (including cost, licensing and usability), not just pixel thinning.
4. Clinician/researcher critique of report usefulness and failures.
5. Written pilot interest or observed time/cost advantage; verify regulatory/privacy and software licensing.

## Owners and system of record
Human owner: project owner. ChatGPT: scientific/product planning and reviews. Codex: implementation. GitHub: code/reproducibility source of truth. Notion: decision history and cross-chat continuity.
