# 08 — University CNN project: teaching, live demonstration and honest pitch

## What the professor assigned

Design a project linked to a dataset or scientific paper, implement a CNN, understand its mathematics and explain the result. **The immediate objective is a working demonstration**, not a full medical research product. No exact final presentation date has been established. On 9 Nov 2026 the priority was to have tangible evidence that development had begun.

## Project pitch for 30–60 seconds

> I'm extending a previous Python research project called MedSkel, where I studied how to extract the centerlines of retinal blood vessels. My CNN project tackles the step before that. Given a photo of the retina, I'll train a small U-Net to mark which pixels belong to blood vessels. The training labels come from the FIVES dataset, which has expert annotations. Then I'll evaluate the predictions against images the model has not trained on and eventually feed the result into my existing measurement code. The goal for class is to understand and test the CNN, not to claim it can diagnose diseases.

## Demo evolution — label each stage accurately

### Day-1 proof: already exists

Show `notebooks/MedSkel_AI_Day1.ipynb`:

1. Installed PyTorch — owner reports version `2.11.0+cpu` in VS Code.
2. Real sample retinal photograph displayed — for illustration only.
3. Small U-Net created; input `(1,3,128,128)` → output `(1,1,128,128)`, 29,761 learnable parameters.
4. Synthetic backprop software test was run in an earlier environment; user had not reported personally reaching it when the package was generated.
5. Optional FIVES loader and tiny-batch training cells are included but **not confirmed executed on real data**.

**Script to say:** 'I implemented the U-Net structure and checked its forward pass. The synthetic optimizer test demonstrates the math/code can run. I have not yet shown real retinal vessel segmentation accuracy.'

### First meaningful CNN training demo — next milestone

Display the following on the same training or validation example, with accurate labels:

```text
Original retinal image | Expert vessel mask | CNN probability/mask | Overlay
```

Show actual training loss history and known dataset split. Explain that training images and evaluation images must be separate. For early tiny-batch training, show it as a debugging step, not a generalization result.

### Final class demonstration — after validation

- Original retinal image.
- Segmentation mask predicted by frozen U-Net.
- Expert annotation overlaid to reveal false positives and missed vessels.
- Per-image and aggregate validation/test Dice, IoU, precision, recall from held-out data.
- A simple centerline and pixel-space length/width/branch table; first with thinning, later optionally MedSkel.
- A fair statement of errors: missed thin vessels, optic-disc artifacts, low-contrast cases, scale limitations.
- One slide of comparison with original Frangi method when feasible.

## A student-friendly explanation of U-Net

**What is the question?** Not 'is the whole image a diseased eye?' Instead it is 'does **each pixel** belong to a blood vessel?'

**Why 3 channels in and 1 out?** The RGB photo carries red, green and blue; the single output channel stores a *vessel score* for every pixel. This is **not grayscale conversion**: a vessel score is learned from labels, not just brightness.

**What is convolution?** A small sliding filter multiplies nearby pixel values by learned numbers and sums them. An early filter may respond to edges or lines; deeper ones learn combined features. Show one 3×3 filter example on the whiteboard.

**What is a feature map?** The array of responses from one filter over the image; 8 learned filters make 8 maps, not 8 new colors.

**What is an encoder?** Several convolution blocks with pooling; spatial resolution shrinks, allowing features to combine wider context.

**What is a decoder?** Upsampling plus convolutions recovers a pixel-sized output. The saved early encoder maps help keep fine details via **skip connections**.

**What are parameters?** Learnable kernel coefficients and biases. The Day-1 CNN has 29,761, but their number isn't an accuracy score; at initialization they are not trained vessel detectors.

**What is loss?** A number measuring how different the CNN's vessel scores are from the known expert mask. BCEWithLogitsLoss is computed from raw logits and targets.

**What is backpropagation?** A method to calculate the effect of each weight on loss. The optimizer changes weights using gradients to reduce future training loss.

**What is overfitting?** The model remembers training masks but fails on unseen retinal photos; that's why we need validation/test separation and genuinely new examples.

**What is Dice?** `2 overlap / (number predicted vessel pixels + number expert vessel pixels)`; a mask overlap measure, not direct proof of correct vessel widths or medical predictions.

## The mathematics to be prepared to explain

1. Tensor shapes at every block: `[batch, channels, height, width]`.
2. 3×3 convolution weight count: `(in_channels × 3 × 3 + bias) × out_channels`. E.g. `3→8` gives 224 parameters.
3. Output size formula and why `padding=1` keeps resolution with kernel size 3.
4. Pooling `128→64→32`, then decoder `32→64→128`.
5. Concatenation skip `32+16=48`, `16+8=24` channels.
6. Sigmoid converting logit to probability and why BCEWithLogitsLoss is used for training.
7. Dice, precision and recall and what false positive/false negative means in a vessel mask.
8. What a gradient and Adam update do, with no claim that one update is full training.

## Simple presentation outline (6–8 slides)

1. The problem: manual retinal vessel tracing, with actual fundus and mask.
2. Why it matters to research, cautiously citing retinal vascular atlas (not a diagnosis claim).
3. Prior MedSkel research: geometry-based centerlines and honest trade-offs.
4. CNN proposal: small U-Net, FIVES annotations, input/output.
5. How U-Net works: encoder, decoder, skip connections, 3×3 filters, pixel logits.
6. Implementation: PyTorch structure, parameter count, pipeline and training setup.
7. Actual results: **only numbers obtained from our executed experiments**; show Dice/IoU when real evaluation completed.
8. Limits and next step: weak vessels, data leakage, units, physical measurement validation.

### Questions professor may ask

- 'Why U-Net, not standard CNN classification?' → We need a per-pixel mask; U-Net combines local details and broader context.
- 'Why expert masks?' → They give the supervised pixel labels for the loss.
- 'Why not train on all 800 images?' → 200 official examples are reserved for final testing; use training-only validation and beware 573-patient source.
- 'Does this predict diabetes or blood pressure?' → No; we segment vessels. Published association/prediction papers are motivation, distinct methods.
- 'Is U-Net original research?' → No, original 2015. Our work applies and studies it, connects with earlier geometry and evaluates measurements.
- 'What does 29,761 mean?' → Number of adjustable weights/biases in our deliberately tiny network, not prediction quality.
- 'Why didn't you just use your old MedSkel?' → It needs a binary vessel mask first; learned segmentation tackles that missing stage.
- 'Are your measurements in millimeters?' → No, pixels until a justified scale calibration is available.
- 'Why is your old geometry not simply better?' → It stabilizes total-length differences between observers but hurts positional agreement and runtime; comparison must be honest.

## Rule for agents who create slides or scripts

Never create 'accuracy 98%' or fake loss curves. If real training is not done yet, show a technical-start slide with honest status. Use plain explanations suitable for a student learning CNN mathematics; never assume the owner can defend generated code without walking through it.
