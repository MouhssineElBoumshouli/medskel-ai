# MedSkel AI — Primary resources and competitor register
Prepared 5 November 2026; links/versions/licenses must be checked again before use.

## Starting code and original mathematical method
- Original code: https://github.com/MouhssineElBoumshouli/medskel
  Inspect README, `medskel/voronoi.py`, `medskel/bisector.py`, `medskel/metrics.py`, `medskel/data.py`, `medskel/chase.py`, `experiments/`, `tests/`.
- Nourddin Saidou, Zineddine, Rhazzaf (2024), *A Novel Method of Skeletonization of Complex Shapes Based on Bisectors*. The paper PDF is already included at `docs/saidou-et-al-2024-a-novel-method-of-skeletonization-of-complex-shapes-based-on-bisectors.pdf` in that repo. Compare the claimed straight skeleton vs medial-axis definitions carefully.

## CNN paper
- Ronneberger, Fischer, Brox (2015), U-Net: https://arxiv.org/abs/1505.04597
  Raw material: U-Net architecture and math, with exact encoder/decoder and skip connections.

## Retinal datasets
- FIVES original paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9352679/
  800 fundus images with expert vessel masks and image-quality assessments. Use official split as appropriate, but verify patient identity and duplicate handling. Original dataset distribution referenced in the paper; convenient public dataset listing: https://huggingface.co/datasets/tyluan/FIVES (CC BY 4.0 as listed; verify raw source license/attribution).
- CHASE_DB1 official Kingston page: https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/
  28 images, 14 children (left/right eyes), two human vessel masks, CC BY 4.0. This source is essential to reproducing original MedSkel observer experiments. The source annotations are *not* independent measurements of physical vessel width.

## Competitor / comparison tools
- AutoMorph (Zhou et al., 2022): https://pmc.ncbi.nlm.nih.gov/articles/PMC9290317/
  GitHub: https://github.com/rmaphoh/AutoMorph
  Automated QC, vessel and artery/vein detection, optic disc and vascular features. Review licenses and exact measurement units/camera calibration before using its outputs.
- VascX (Vargas Quiros et al., 2026): https://arxiv.org/abs/2602.08580
  Current code: https://github.com/Eyened/retinalysis-vascx
  Research stack includes vessel masks, graph tracking, calibre/CRAE/CRVE/AVR, tortuosity, bifurcation and region-aware features. IMPORTANT: current GitHub README/PyPI describe PolyForm Noncommercial 1.0.0 licensing, while a publication describes an earlier open-source license. Verify license for EXACT version and components. Benchmarking may be allowed while commercial code reuse may require permission.
- For camera/resolution problems: https://www.nature.com/articles/s41598-026-44196-y — different cameras and image resolutions change automated measurement behavior; use as reason to validate cross-device robustness.

## Retinal biomarker definitions
- CRAE = central retinal arteriolar equivalent; CRVE = venular equivalent; AVR = CRAE/CRVE, according to a defined vessel-selection/optic-disc region/formula and appropriate calibration context. These are *not* the current MedSkel mean radius or generic vessel-width output.
- Retinal vessel width and tortuosity may correlate with disease risk factors at population level, but image-derived biomarkers do not by themselves diagnose a patient.

## Licenses and ethics
- Record dataset paper, distribution URL, stated license, attribution, allowed use, access date, subject/annotation considerations and any redistribution restrictions.
- No identifiable patient files stored in GitHub or uploaded to assistants without prior approvals.
- Before public clinical deployment, seek professional review of intended-use medical-device classification and health-data laws in target jurisdictions.

## Resources to add to ChatGPT project
1. Upload `02_PROJECT_CHARTER.md`, `03_EXPERIMENT_PROTOCOL.md`, `04_RESEARCH_RESOURCES.md`, `05_DECISION_LOG.md` individually.
2. Optionally upload the 2024 Saidou PDF from your original repository and U-Net original paper PDF, after checking available source slots.
3. Use live GitHub connector for code rather than uploading a frozen repository ZIP that goes stale.
4. Keep large datasets in the controlled development environment, not in ChatGPT project sources.
