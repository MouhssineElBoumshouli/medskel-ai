# 06 — Original research, medical motivation and existing alternatives

**Source check date: 9 November 2026.** Prefer original publications, dataset creators and official code repositories. Check licenses, code versions and data availability again at time of use. Links are to sources, not claims that we reproduced their experiments.

## A. Scientific foundations (read the originals)

### 1. Nourddin Saidou et al. — bisector-based skeletonization (2024)

- Title: *A Novel Method of Skeletonization of Complex Shapes Based on Bisectors*.
- Publisher DOI: https://doi.org/10.1142/S2972370123500083
- Source PDF in original repo: https://github.com/MouhssineElBoumshouli/medskel/blob/main/docs/saidou-et-al-2024-a-novel-method-of-skeletonization-of-complex-shapes-based-on-bisectors.pdf
- MedSkel implementation: https://github.com/MouhssineElBoumshouli/medskel
- **Why relevant:** Geometric centerline extraction inspired by boundary angle bisectors and polygon simplification. The original project implemented and investigated this, not machine learning.
- **Scientific caution:** Straight skeleton and medial axis are mathematically different in general for concave polygons. The standalone wavefront implementation is partial (split events not implemented). Do not claim it is a full or exact implementation for all shapes merely because hand-derived convex tests match.

### 2. Ronneberger, Fischer, Brox — U-Net (2015)

- Title: *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- Paper: https://arxiv.org/abs/1505.04597
- **Why relevant:** Original encoder/decoder segmentation architecture with skip connections. Relevant class mathematics: convolutional filters, feature maps, spatial downsampling, upsampling, concatenation, per-pixel prediction, gradient descent.
- **Scientific caution:** The Day-1 29,761-parameter model is deliberately **smaller/simpler than original U-Net**. Identify it as 'small U-Net inspired by Ronneberger et al.', not a faithful replication with identical layers or reported benchmark performance.

### 3. Jin et al. — FIVES (2022)

- Title: *FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation*.
- Paper: https://doi.org/10.1038/s41597-022-01564-3
- Open full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC9352679/
- Dataset publisher: https://doi.org/10.6084/m9.figshare.19688169
- **Why relevant:** Original high-resolution color retinal images with manual binary vessel annotations and image quality scores; 800 images, 573 participants, paper-recommended 600 train / 200 test.
- **Scientific caution:** Public images are not equivalent to independent participants. Distinguish the paper's license from underlying dataset license and check exact raw data rights.

### 4. CHASE_DB1 — independent observer masks (2012)

- Official dataset: https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/
- **Why relevant:** 28 retinal photographs (both eyes of 14 children), annotated independently by two human observers. This allowed original MedSkel to test sensitivity of measurements to annotation differences.
- **Scientific caution:** Observer disagreement is **not** independent anatomical truth. The two eyes of a child are related; subject-aware statistics required. Images are of children and differ from FIVES data, so external evaluation may expose significant domain shift.

## B. Retinal measurements and associations are real research, not product claims

### 5. Yusufu et al. — retinal vascular atlas (2025)

- Title: *Population-Based Normative Reference for Retinal Microvascular Atlas*.
- Paper: https://doi.org/10.1016/j.xops.2025.100723
- Free full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC11950772/
- Study type: **cross-sectional association and normative-reference study** among **10,151** selected healthy UK Biobank participants.
- Method: Established retinal vascular analysis software extracted **114** vascular features, including arterial and venous calibre, density, complexity, branching angle and tortuosity, in defined retinal regions. Compared patterns with age, systolic blood pressure, BMI, HbA1c and intraocular pressure.
- **What it supports:** Vessel measurements can be scientifically informative and associated with health factors in selected populations.
- **What it does not support:** Deriving exact blood pressure from our lengths/widths, diagnosing diabetes or establishing clinical outcomes for MedSkel AI. Medical associations from one population need validation across populations and imaging methods.

### 6. Poplin et al. — risk factors predicted directly from photographs (2018)

- Title: *Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning*.
- Original paper: https://doi.org/10.1038/s41551-018-0195-0
- Google Research page: https://research.google/pubs/predicting-cardiovascular-risk-factors-in-retinal-fundus-photographs-using-deep-learning/
- Studied training photographs from **284,335** patients, external sets of **12,026** and **999**, and reported systolic BP mean absolute error **11.23 mmHg**, as well as other risk factor predictions.
- **Crucial difference:** Prediction was learned directly from fundus photographs, **not a simple regression from the small set of geometrical vessel measurements currently computed by MedSkel**. This is **not** evidence that a MedSkel output directly estimates blood pressure or gives a clinical-grade reading.

### 7. Why these articles matter for the class talk

A simple one-sentence motivation: 'Researchers study retinal vessel patterns because these patterns are associated with eye and general health factors. Before studying those patterns, we need a reliable way to find the vessels. My CNN tackles that first technical step.'

Do **not** spend the first class presentation proposing a 'diabetes scanner'. The defensible original contribution is learning to segment vessels and testing whether that improves downstream geometry, not inventing another risk score.

## C. Competitive landscape: you do not have a novel category yet

### 8. AutoMorph — established retinal image analysis pipeline

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9290317/
- Original GitHub: https://github.com/rmaphoh/AutoMorph
- Capabilities: image-quality assessment, vascular segmentation, artery/vein labeling, optic disc handling, automated retinal vascular features and research workflows.
- Code's core license was previously observed as Apache 2.0; re-check current upstream and each model weight, dependency, dataset and redistributed component before reuse.
- **Implication:** 'A pipeline that segments retinal vessels and measures them' is **not novel by itself**. Comparison must focus on concrete reproducibility, calibration, accuracy, correction workflows or speed/quality tradeoffs.

### 9. VascX / retinalysis-vascx (2025–2026)

- Current code: https://github.com/Eyened/retinalysis-vascx
- 2026 software preprint: https://arxiv.org/abs/2602.08580
- Project/PyPI: https://pypi.org/project/retinalysis-vascx/
- Features documented in public software: fundus vessel and artery/vein segmentation, structural graph analyses, central retinal vascular equivalents, tortuosity, density, branching and other biomarkers; subsequent versions added newer segmentation models, ARIC-style equivalents, browser demo and repeatability documentation.
- **License warning:** Public PyPI listing for `retinalysis-vascx` **2.3.1 (29 Oct 2026)** says **PolyForm Noncommercial 1.0.0**. Some older related model repositories describe *different* code and weight licenses, including noncommercial data/models. Do not confuse them, and never incorporate software/weights for a future commercial system without precise license analysis/permission.
- **Implication:** Competitor can already extract many measurements; its capabilities, failure conditions, reproducibility, license and usability become benchmarks, not features we can claim no one has built.

### 10. Criteria for a fair competitor comparison

- Compare identical, permitted test images and note if an incumbent's training data may overlap with test datasets.
- Define exact measurement formulas and optic-disc/artery/vein region requirements; do not compare unlike units.
- Use comparable quality control, camera scaling, hardware and runtime settings.
- Evaluate output failures and manual correction effort, not just overall Dice.
- Check citation, source license, model weights rights and derivative-code constraints separately.
- A commercial hypothesis is only supported by *verified* researcher workflow pain and measured advantage, not GitHub stars or novelty rhetoric.

## D. How to use the bibliography in further research

Research agent should create a living table with `source, DOI/URL, version, population, images/patients, training/test split, task, metric, limitations, license, date checked, relevance`. Preserve quotations and results with attribution. Distinguish original-paper evidence from author's interpretations and our hypotheses. For disease/BP discussions, highlight whether a paper measured **association**, predicted an outcome, or established a clinically validated intervention.
