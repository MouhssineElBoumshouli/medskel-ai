# 03 — CNN model: implementation contract, math and reproducibility

## Goal

Build a **small, understandable, trained U-Net for binary retinal vessel segmentation**, then connect its output to geometry for a research demonstration. PyTorch is the preferred stack. Do not prematurely substitute a giant pretrained model or a framework that hides the architecture from a student expected to explain it.

## Input, target, output

- Input color fundus photograph `x`: array height × width × RGB, converted to float PyTorch tensor **`[B,3,H,W]`**, values normally scaled to `[0,1]` (any further normalization must be learned from the *training* set and recorded).
- Ground-truth mask `y`: **`[B,1,H,W]`**, 0=background/non-vessel, 1=vessel. Ensure masks are actually binary after decoding; check inverted or anti-aliased labels.
- U-Net output `z`: **`[B,1,H,W]`** real-valued *logits*. No sigmoid in the model if training with `BCEWithLogitsLoss`.
- Vessel probability: `p = sigmoid(z) = 1/(1+exp(-z))`; apply a threshold `t` tuned only on validation (not test) to obtain binary predicted mask `p >= t`.
- For presentation: overlay the predicted mask on the original image; add a legend indicating predicted vessel pixels. Do not use a threshold tuned by looking at individual test images.

## Day-1 model implementation (as it exists)

`notebooks/MedSkel_AI_Day1.ipynb` defines `SmallUNet` with two encoder blocks, one central bridge, two decoder blocks and skip connections. Three-convolution-channel map summary:

| Stage | Shape for 128×128 | What happens |
|---|---|---|
| Input | `(1,3,128,128)` | RGB photo |
| `enc1` | `(1,8,128,128)` | Two padded 3×3 convs with ReLU |
| Pool | `(1,8,64,64)` | 2×2 max pooling, stride 2 |
| `enc2` | `(1,16,64,64)` | Learn larger scale patterns |
| Pool | `(1,16,32,32)` | Downsample once more |
| Bridge | `(1,32,32,32)` | Higher-level context |
| Interpolate+cat `enc2` | `(1,48,64,64)` | Upsample, restore saved details |
| `dec2` | `(1,16,64,64)` | Feature processing |
| Interpolate+cat `enc1` | `(1,24,128,128)` | Restore detailed spatial features |
| `dec1` | `(1,8,128,128)` | Feature processing |
| Output `1x1 Conv` | `(1,1,128,128)` | One logit/pixel |

29,761 trainable parameters confirmed in notebook. Count includes learnable kernels and biases. With shape-preserving padding and divisible-by-four spatial sizes, the decoder returns original H×W; use explicit interpolation sizes to avoid off-by-one for arbitrary shapes. This tiny network is for debugging and CPU training, not a presumed optimal model.

## Basic mathematics student must understand

### Convolution

Each output feature is a weighted sum of neighboring input pixels, followed by a bias:

`Y[o,i,j] = b[o] + sum_c sum_u sum_v W[o,c,u,v] * X[c,i+u,j+v]`.

For a 3×3 convolution from 3 RGB channels to 8 feature channels, the parameter count is `8 * (3*3*3 + 1) = 224`. The `(3,3)` defines the spatial window; output channels are **learned feature maps**, not RGB colors. Deep layers learn increasingly useful combinations of patterns.

Convolution output size per dimension with kernel K, padding P, stride S, dilation D:

`floor((H + 2P - D*(K-1) - 1)/S + 1)`.

For K=3, P=1, S=1, D=1 this preserves H×W.

### ReLU

`ReLU(x) = max(0,x)`. It enables non-linear combinations and helps learn more than a single linear filter.

### Pooling

2×2 max pooling with stride 2 keeps the strongest response in each window and reduces spatial dimensions. This increases effective spatial context but can hide thin vessels. A `128×128` feature map becomes `64×64`, then `32×32`.

### Skip connections

The encoder's early features (`a` and `b`) are **concatenated** channelwise with decoder features at matching resolution. Example: `32` upsampled channels + `16` encoder channels = `48` input channels to `dec2`. Skip connections are important for fine edges, but do not guarantee recovery of vessels already destroyed by input resampling.

### Sigmoid, binary cross-entropy and Dice

For pixel label `y` in `{0,1}`, predicted vessel probability `p`, BCE is `- [y log(p) + (1-y) log(1-p)]`. `BCEWithLogitsLoss(z,y)` computes the same loss stably from logits without first explicitly calling sigmoid.

Dice similarity at the mask level:

`Dice = (2*TP) / (2*TP + FP + FN)` (define empty-mask case). A smooth training Dice loss can be `1 - (2*sum(p*y) + eps)/(sum(p)+sum(y)+eps)`, differentiable in probabilities.

Vessel pixels are relatively rare, so a background-heavy prediction can look good under naive accuracy. Start with BCE as a sanity check, then consider `BCE + soft Dice` for real training; record weighting and run controlled comparisons. Do not silently report a training Dice as a test Dice.

### Backpropagation and optimizer

A forward pass makes `z = f_theta(x)`, loss `L(theta) = Loss(z,y)` compares to target mask. `loss.backward()` computes derivatives `dL/dtheta`, and `optimizer.step()` updates learnable parameters, e.g. Adam. A single optimizer step checks plumbing, **not successful learning**. Show loss decreasing *across multiple train steps* and evaluate on images never used to tune the model.

## Recommended implementation modules [PROPOSED — new dedicated CNN repository]

This **separate `medskel-ai` repository** owns CNN training, datasets, evaluation and the eventual end-to-end demo. The original [`medskel`](https://github.com/MouhssineElBoumshouli/medskel) stays intact and is referenced for scientifically established geometry and its results.

```text
medskel-ai/
  CLAUDE.md                    # agent operating rules
  docs/                        # detailed research and task documents
  notebooks/MedSkel_AI_Day1.ipynb # initial teaching notebook; not real FIVES training
  src/medskel_ai/              # to be created on an approved feature branch
    __init__.py
    data.py                    # FIVES image/mask pairs and split integrity
    model.py                   # small U-Net
    train.py                   # reproducible training and validation
    evaluate.py                # per-image metrics and evaluation
    predict.py                 # probability/mask inference
    visualize.py               # paired images, overlays and class figures
  tests/                       # model shapes, data pairing, training smoke, metrics
  configs/                     # explicit run configuration, when needed
```

This layout is a **proposal**, not a claim these code modules already exist. Do not copy the old geometry package or its result tables into this new project as if it were a verified dependency. First train the U-Net. Then, if geometry integration is justified, choose a clean pinned dependency, adapter, or extracted package and test equivalence with the original outputs. Refer to `docs/11_REPOSITORY_BOUNDARIES.md`.

## Development stages and acceptance tests

### A. Data and shape smoke tests

- Given a known image and label pair, load RGB `[3,H,W]` and binary `[1,H,W]` with identical spatial sizes.
- Verify nearest-neighbor mask resizing preserves `{0,1}`; verify augmentation flips/rotations align image and mask.
- Assert masks are neither unexpectedly all-black nor all-white; record foreground vessel fraction.
- Assert every dataset path is within a known train/validation/test manifest; same parent image never appears across splits.
- Given `[2,3,128,128]`, U-Net returns `[2,1,128,128]`; output is finite.

### B. Optimization smoke test

- Take **two real FIVES training pairs** (never test). Backprop finite BCE loss, confirm nonzero gradients, save summary of steps. Short overfit test: can the model substantially reduce loss on the same tiny batch? If not, fix the training code before training all data. Do not use this tiny-batch result as generalization evidence.
- CPU run should be small; choose test resolution based on available resources. Do not assume training full 2k×2k FIVES images is feasible on CPU.

### C. Clean baseline run

- Keep official 200-image FIVES test portion sealed.
- Create training-only validation with documented subject-group constraints and limitation. If patient group mapping is missing, do not pretend proven patient independence.
- Record seed, Python/PyTorch versions, model parameter count, hardware, optimizer, LR, augmentation, effective resolution, epochs and all losses.
- Save best checkpoint by a predeclared **validation** criterion and save the chosen validation threshold. Keep a plain JSON config beside checkpoint.
- Record per-image validation Dice/IoU/precision/recall; handle zero-division.
- Preserve training curves and a small fixed set of representative *validation* overlays, including failures.

### D. Locked independent evaluation

- Only after model/threshold selection, evaluate on never-used official FIVES test images. Report per-image values and uncertainty. FIVES may have repeated-patient images; disclose if public metadata cannot ensure group separation.
- Evaluate on CHASE_DB1 as truly external images if the model was not trained/tuned using CHASE. Use correct image dimensions/FOV and note domain shift and subject clustering.
- Include comparison to a trivially simple baseline, not just a pretty U-Net prediction.

## Image quality, hardware and optimization caveats

- Initial `128×128` whole-image resizing can erase capillaries, producing misleading thin-vessel metrics. The class smoke test can use 128, but later test 256/512 or patches and confirm actual retained vessels.
- For CPU-only machine, start small with batch size 1–4 and no large UI. If time or compute become blocking, inspect actual performance and consider an approved external accelerator; do not spend money or upload health data automatically.
- Train augmentations on images *and masks* consistently for geometry; photometric changes only to image. Validation/test transforms deterministic.
- Avoid mask leakage by generating no labels from the same Frangi algorithm being benchmarked as a baseline.
- Write unit tests around data, metrics and predictions before producing polished interface.
- Explain sigmoid output and threshold to user; output alone is not clinical information.
