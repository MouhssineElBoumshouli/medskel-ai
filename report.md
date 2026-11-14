# Retinal vessel segmentation with a small U-Net on the FIVES dataset

**Author:** Mouhssine El Boumshouli — CNN class project — 14 November 2026

**Code:** `MouhssineElBoumshouli/medskel-ai`, branch `feature/unet-fives`

---

## Abstract

We train a plain U-Net (7.85 million parameters) to mark blood vessels in colour photographs of the retina. We use the public FIVES dataset: 800 photographs of 2048 x 2048 pixels with expert vessel masks. We keep the full resolution by training on random 512 x 512 crops and by predicting whole images tile by tile. Training took 51 minutes on a laptop GPU with 4 GB of memory. The decision threshold was chosen on validation images only. On the 200 official test images, evaluated once, the model reaches a mean Dice of **0.875** (median 0.907), precision 0.909 and recall 0.856. The model works well on most images and fails on a few, mostly in the glaucoma group. The result is a class demonstration. It is not a clinical tool, and patient overlap between the dataset parts cannot be ruled out.

## 1. Introduction

Blood vessels of the retina can be photographed directly. Their shape is studied in research on eye disease and on general vascular health. Before anyone can measure a vessel, a program must decide which pixels belong to vessels. This is a **binary segmentation** task: one yes/no answer per pixel.

The task is hard for simple methods. Vessels are thin, sometimes only one or two pixels wide. They have low contrast. And they are rare: in our training masks only 7.5% of the pixels are vessel.

The goal of this project is educational: build the smallest U-Net that is still a real model, train it on real expert labels, evaluate it honestly, and understand every step. The mathematics is explained in `docs/explanation.md`.

## 2. Data

**FIVES** (Jin et al., 2022) contains 800 colour fundus photographs with pixel-level vessel masks drawn by experts. There are four groups of 200 images: AMD (A), diabetic retinopathy (D), glaucoma (G) and normal (N). The authors define a split of 600 training and 200 test images. The group letter is part of each file name.

We checked every file of our local copy with a script (`python -m medskel_ai.data`, result in `results/data_audit.json`):

| Check | Result |
|---|---|
| Image/mask pairs with identical names | 600 train, 200 test |
| Images per group | 150 per group in train, 50 per group in test |
| Image size | all 2048 x 2048, masks the same size |
| Mask values | only 0 and 255 in all 800 masks |
| Vessel share of pixels (train) | mean 7.5%; A 7.3%, D 6.9%, G 6.7%, N 9.0% |
| Completely black masks | 2 in train (`447_G`, `448_G`), 0 in test |
| Byte-identical photographs | 6 pairs: 5 inside train, 1 inside test, 0 between train and test |

Two findings changed our plan:

- The two **empty masks** are label errors. We excluded these two images from training and validation.
- The **identical photographs** are the same file stored under two names. Their masks are not identical, so the same photo was annotated twice. In our first split, one such pair was divided between training and validation. This would leak information. We changed the split so that copies always stay in the same part. A unit test covers this.

**Split.** The official test set is kept as it is. From the 598 remaining training images we took 88 for validation (22 per group, random with seed 42) and kept 510 for training. The list is in `results/split_seed42.json`.

**Important limit.** The FIVES paper states that the 800 images come from 573 patients. The public data contains no patient identifiers. Different photographs of the same patient can therefore be in different parts of our split and of the official split. We cannot check this and cannot fix it. Our scores are scores on *unseen images*, not proven scores on *unseen patients*.

## 3. Method

### 3.1 Preprocessing

Images are read as RGB and divided by 255, so values are between 0 and 1. Masks are converted to exactly 0 and 1. Nothing is resized. If a mask ever has to be resized (only for display), nearest-neighbour interpolation is used so that labels stay 0/1.

For training we cut two random 512 x 512 crops from each photograph. Each crop is randomly flipped and rotated by a multiple of 90 degrees, with the image and the mask transformed identically. Brightness and contrast are changed slightly on the image only. Validation during training uses four fixed crops around the image centre, with no randomness.

### 3.2 Model

The model is a standard U-Net (Ronneberger et al., 2015) written in plain PyTorch (`src/medskel_ai/model.py`).

- **Encoder:** four blocks with 32, 64, 128 and 256 channels. Each block is two 3 x 3 convolutions, each followed by batch normalisation and ReLU. After each block, 2 x 2 max pooling halves the size.
- **Bottleneck:** one block with 512 channels at 1/16 of the input size.
- **Decoder:** four steps. Each step upsamples by 2 (bilinear), concatenates the encoder feature map of the same size (skip connection), and applies a block.
- **Head:** a 1 x 1 convolution that gives one logit per pixel.

For a 512 x 512 crop the tensor goes from `[3, 512, 512]` down to `[512, 32, 32]` and back to `[1, 512, 512]`. The model has **7,849,601** trainable parameters. No pretrained weights are used.

### 3.3 Loss and optimisation

The loss is the sum of two terms with equal weight:

- **Binary cross-entropy** on the logits (`BCEWithLogitsLoss`), averaged over pixels.
- **Soft Dice loss**: `1 − (2·Σ p·y + 1) / (Σ p + Σ y + 1)`, where `p` is the predicted probability and `y` the label.

BCE gives a stable gradient for every pixel. Dice focuses on the overlap of the rare vessel class. We did not compare this choice with BCE alone.

| Setting | Value |
|---|---|
| Optimiser | Adam, learning rate 0.001, lowered to 0 with a cosine curve |
| Batch | 4 crops (2 photographs x 2 crops) |
| Mixed precision | yes (16-bit where safe) |
| Time budget | 50 minutes |
| Seed | 42 |
| Hardware | NVIDIA RTX 3050 Laptop GPU, 4 GB; peak GPU memory used by PyTorch 2.0 GB |
| Software | Python 3.11.9, PyTorch 2.14.1 (CUDA 12.6 build) |

After every epoch we compute the Dice score on the fixed validation crops (threshold 0.5) and save the weights if it is the best so far.

### 3.4 Prediction on whole images

A whole image is cut into 512 x 512 tiles with an overlap of 64 pixels (25 tiles for a FIVES image). Each tile is predicted. Probabilities are averaged where tiles overlap. A pixel is called vessel if its probability is at least the threshold.

### 3.5 Evaluation protocol

1. Train, and choose the best epoch with validation crops.
2. Predict the 88 **whole** validation images. Try thresholds 0.10, 0.15, …, 0.90. Keep the one with the best mean Dice.
3. Only then predict the 200 official test images, **once**, with that threshold. The script refuses a second run.

Metrics are computed **per image** and then averaged, so each image counts equally. All pixels of the image are used (no field-of-view mask). Dice, IoU, precision, recall, specificity and accuracy are defined in `docs/explanation.md`, section 9.

## 4. Results

### 4.1 Training

Training ran 38 epochs in 51.2 minutes. The training loss fell from 1.00 to 0.16, the validation loss from 0.83 to 0.21. The best validation Dice on fixed crops was 0.878 at epoch 32. The validation loss did not rise again at the end, and the gap to the training loss stayed small. We see no sign of strong overfitting.

![Training curves](results/curves.png)

*Figure 1. Left: loss on training crops and on validation crops. Right: Dice on validation crops. Made from `results/history.csv`.*

### 4.2 Threshold

Mean Dice on the 88 whole validation images was highest at threshold **0.40** (0.8985). The curve is flat: every threshold from 0.30 to 0.45 gives a value within 0.001 of the best. At threshold 0.40 the validation scores are Dice 0.899, IoU 0.827, precision 0.918, recall 0.887.

### 4.3 Official test set

**Table 1. Test results, 200 images, threshold 0.40. Mean ± standard deviation over images.**

| Metric | All (200) | A – AMD (50) | D – DR (50) | G – glaucoma (50) | N – normal (50) |
|---|---|---|---|---|---|
| Dice | **0.875 ± 0.114** | 0.908 ± 0.055 | 0.876 ± 0.082 | 0.828 ± 0.190 | 0.887 ± 0.052 |
| IoU | 0.791 ± 0.136 | 0.836 ± 0.083 | 0.787 ± 0.116 | 0.739 ± 0.206 | 0.801 ± 0.078 |
| Precision | 0.909 ± 0.063 | 0.925 ± 0.049 | 0.925 ± 0.055 | 0.901 ± 0.089 | 0.886 ± 0.039 |
| Recall | 0.856 ± 0.144 | 0.898 ± 0.087 | 0.840 ± 0.118 | 0.794 ± 0.218 | 0.892 ± 0.080 |
| Specificity | 0.993 ± 0.005 | 0.994 ± 0.005 | 0.995 ± 0.004 | 0.996 ± 0.003 | 0.989 ± 0.004 |
| Accuracy | 0.986 ± 0.006 | 0.987 ± 0.006 | 0.986 ± 0.006 | 0.988 ± 0.003 | 0.982 ± 0.006 |

Other summary numbers: median Dice 0.907, best image 0.956, worst image 0.031. Ten images have a Dice below 0.7. If all pixels of all test images are pooled together first, Dice is 0.899; this number hides the bad images and is not our main result.

### 4.4 Why accuracy is misleading

**Table 2. Our model against a "predict nothing" baseline on the test set.**

| Model | Accuracy | Dice | Recall |
|---|---|---|---|
| All-black mask (no vessels at all) | 0.929 | 0.000 | 0.000 |
| Our U-Net | 0.986 | 0.875 | 0.856 |

About 93% of the pixels are background. A model that finds nothing is therefore "93% accurate". Accuracy mostly rewards the easy background. Specificity has the same problem: it is above 0.99 for every group and says almost nothing. Dice, IoU, precision and recall look at the vessel class and separate a useless model from a useful one. They are our main metrics.

### 4.5 Where the model fails

- **Recall is lower than precision** (0.856 against 0.909). The model seldom invents vessels, but it misses about one vessel pixel in seven. We expect these to be mostly thin, faint vessels, but we did not measure thin vessels separately.
- **Glaucoma images are the weakest and the most variable** (Dice 0.828 ± 0.190). The four worst test images are all from this group. In the two worst (`122_G`, Dice 0.03; `123_G`, Dice 0.11) the expert mask itself contains very few vessel pixels (0.7% and 1.7% of the image, against a usual 7%), and the model predicts almost nothing (0.03% and 0.12%). These may be low-quality photographs. We have not verified this by eye.
- **Normal images have the lowest precision** (0.886) although their recall is high. The model marks slightly more than the expert there.
- On validation, one image (`174_D`) has Dice 0.15 although the model predicts about the same *amount* of vessel as the expert (8.1% against 8.5%). Prediction and mask do not overlap. A possible reason is a mask that does not belong to this photograph. This is a guess and was not checked.

Six example figures (image | expert mask | prediction | overlay) were saved to `runs/unet_fives/test_examples/`. They were chosen by a fixed rule, not by hand: the worst image, the best image, and the images at 20%, 40%, 60% and 80% of the Dice ranking. They are not in the Git repository, because the project rule is to keep dataset images out of Git.

## 5. Discussion and limits

The model does what a small U-Net should do: it finds most of the vessel tree on most images after less than one hour of training on a laptop. The numbers must be read with these limits:

1. **Patient overlap is possible.** 800 images come from 573 patients and no identifiers are published. The scores can be optimistic compared with truly new patients.
2. **One run.** One seed and one training run. We have no error bar for the training itself.
3. **One dataset.** No external test (for example CHASE_DB1). Other cameras and populations will probably give lower scores.
4. **Labels are not perfect.** We found two empty masks and duplicated photographs with different annotations. The expert mask is a reference, not biological truth.
5. **No field-of-view mask.** The dark border counts as background. This raises accuracy and specificity.
6. **Model selection used centre crops**, not whole images. The retina border is under-represented there.
7. **No comparison.** We did not train another model or a classical filter on the same split. We make no claim that this model is better than any published method, and we do not compare our Dice with numbers from papers, because their protocols differ.
8. **Segmentation only.** No vessel width, length or branching is measured. Nothing in this project supports any medical statement about a patient.

## 6. Conclusion

A plain U-Net with 7.85 million parameters, trained for 51 minutes on full-resolution crops of FIVES, reaches a mean Dice of 0.875 on the 200 official test images, with the threshold fixed beforehand on validation data. The typical image is segmented well (median Dice 0.907). A small number of images, mainly glaucoma cases with sparse masks, fail badly. Accuracy is not informative for this task: an empty prediction already reaches 93%.

Next steps, in order: look at the worst images by eye, repeat the training with several seeds, measure thin vessels separately, and test on an external dataset.

## Reproducing the results

```powershell
.\.venv\Scripts\python.exe -m pytest                                # 22 tests, synthetic data
.\.venv\Scripts\python.exe -m medskel_ai.data                       # data audit + split
.\.venv\Scripts\python.exe -m medskel_ai.train --minutes 50         # training
.\.venv\Scripts\python.exe -m medskel_ai.evaluate --split val       # threshold
.\.venv\Scripts\python.exe -m medskel_ai.evaluate --split test      # final test, once
.\.venv\Scripts\python.exe app.py                                   # demo
```

Result files: `results/history.csv`, `results/val_metrics.json`, `results/test_metrics.json`, `results/test_per_image.csv`. Training on a GPU with mixed precision is not bit-for-bit repeatable, so a new run will give slightly different numbers.

## References

1. O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015. https://arxiv.org/abs/1505.04597
2. K. Jin et al. *FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation.* Scientific Data 9, 475 (2022). https://doi.org/10.1038/s41597-022-01564-3 — data: https://doi.org/10.6084/m9.figshare.19688169
3. D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. https://arxiv.org/abs/1412.6980
4. F. Milletari, N. Navab, S.-A. Ahmadi. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation* (Dice loss). 3DV 2016. https://arxiv.org/abs/1606.04797
