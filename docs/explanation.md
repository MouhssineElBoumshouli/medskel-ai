# How our CNN works — explained simply

This file is for you, the student. Short sentences. Small numbers. Every idea points to the code line where it lives.

The task: we give the network a colour photograph of the retina. For **every pixel** it answers one question: *is this pixel part of a blood vessel?* This is called **binary segmentation**.

- Input: a tensor `[3, H, W]`. 3 = red, green, blue. Values from 0 to 1.
- Target: the expert mask `[1, H, W]`. 1 = vessel, 0 = background.
- Output: a tensor `[1, H, W]` of **logits**. One number per pixel.

---

## 1. Convolution

A convolution slides a small window (the **kernel**) over the image. At each position it multiplies the numbers under the window by the kernel numbers and adds everything up.

Tiny example. The image has a dark vertical line (value 2) on a bright background (value 9):

```text
image (4 x 4)        kernel (3 x 3)
9 9 2 9               1 -2  1
9 9 2 9               1 -2  1
9 9 2 9               1 -2  1
9 9 2 9
```

Put the kernel on the top-left 3 x 3 window. Each row gives `1·9 − 2·9 + 1·2 = −7`. Three rows: `−21`.

Move one step right. Each row gives `1·9 − 2·2 + 1·9 = 14`. Three rows: `42`.

```text
output (2 x 2)
-21  42
-21  42
```

The output is large (42) exactly where the dark line sits in the **middle** of the window. So this kernel is a "dark vertical line detector". A vessel in a retina photo is a dark line. That is why convolutions fit this task.

Important points:

- **We do not write the kernel numbers. The network learns them.** They are the *weights*.
- One layer has many kernels. Each kernel makes one output **channel** (a "feature map"). Our first layer has 32 kernels, so it makes 32 channels.
- With a 3 x 3 kernel and **padding 1** (a border of zeros around the image), the output keeps the same height and width. Our example had no padding, so 4 x 4 became 2 x 2.
- Parameter count of our first convolution: `3 input channels × 32 kernels × 3 × 3 = 864` weights.

Code: `nn.Conv2d(..., kernel_size=3, padding=1)` in `DoubleConv`, file `src/medskel_ai/model.py`.

## 2. ReLU

`ReLU(x) = max(0, x)`. Negative numbers become 0. Positive numbers stay.

On our example: `[-21, 42]` becomes `[0, 42]`. Meaning: "no line here" → 0, "line here" → 42.

Why do we need it? A convolution is only sums and multiplications (it is *linear*). Two linear steps in a row are still just one linear step. ReLU breaks this. With ReLU between layers, the network can build complex shapes from simple ones.

**Batch norm** sits between the convolution and the ReLU. It rescales each channel so its values are centred around 0 with a similar spread. This keeps the numbers in a healthy range and makes training faster and more stable.

## 3. Max pooling

Pooling makes the picture smaller. We cut it into 2 x 2 blocks and keep the **largest** value of each block.

```text
1 3 | 2 0
5 2 | 1 1          5 2
----+----    ->    4 9
0 4 | 9 6
1 2 | 3 7
```

Height and width are halved: 512 → 256 → 128 → 64 → 32 in our network.

Why? After pooling, a 3 x 3 kernel covers a larger part of the original photo. So deeper layers "see" more context. They can tell a vessel from the edge of the optic disc. The cost: fine position detail is lost. A vessel that is 1 pixel wide can vanish.

## 4. Upsampling

The answer must have the same size as the input. So the second half of the network makes the picture bigger again. We use **bilinear upsampling**: new pixels are blends of their neighbours.

One row as example: `[2, 6]` becomes `[2, 3, 5, 6]`. The new values 3 and 5 lie between 2 and 6.

Upsampling invents no new detail. It only stretches. The detail comes back through the skip connections.

## 5. Skip connections — the "U" in U-Net

Left side of the U (**encoder**): convolutions + pooling. The picture gets smaller. The network learns *what* is there.

Right side (**decoder**): upsampling + convolutions. The picture gets bigger. The network decides *where* exactly.

A **skip connection** copies the feature map from the left side and **concatenates** it with the upsampled map of the same size on the right side. Concatenate = stack the channels.

Example from our model: the upsampled map has 512 channels. The saved encoder map has 256 channels. Together: `512 + 256 = 768` channels go into the next block.

Why? The deep path knows "there is a vessel around here" but its position is blurry. The saved map has sharp edges but little understanding. Together they give a sharp and correct mask.

The full shape table is at the top of `src/medskel_ai/model.py`. Our model has **7,849,601** trainable parameters.

## 6. Sigmoid: from logit to probability

The network outputs a **logit** `z` per pixel. It can be any number. The sigmoid squeezes it into 0…1:

`p = 1 / (1 + e^(−z))`

| logit z | probability p |
|---|---|
| −3 | 0.047 |
| 0 | 0.500 |
| 2 | 0.881 |

Then we apply a **threshold** `t`: the pixel is called "vessel" if `p ≥ t`. We choose `t` on validation images only.

The sigmoid is **not** inside the model. The loss applies it itself in a numerically safer way.

## 7. The loss: how wrong is the network?

The loss is one number. Small = good. Training means: change the weights so the loss gets smaller.

### Binary cross-entropy (BCE)

For one pixel with true label `y` (0 or 1) and predicted probability `p`:

`BCE = −[ y·ln(p) + (1−y)·ln(1−p) ]`

- True vessel (`y=1`), we say `p=0.9`: `−ln(0.9) = 0.105`. Small. Good.
- True vessel (`y=1`), we say `p=0.1`: `−ln(0.1) = 2.303`. Large. We were confident and wrong.
- True background (`y=0`), we say `p=0.2`: `−ln(0.8) = 0.223`.

BCE punishes confident mistakes very hard. The final BCE is the average over all pixels.

### Soft Dice

Dice measures the **overlap** between prediction and truth: `Dice = 2·overlap / (size of prediction + size of truth)`. 1 = perfect, 0 = nothing in common.

"Soft" means we use the probabilities, not the 0/1 decisions. Four pixels:

```text
truth  y = [1,   1,   0,   0  ]
pred   p = [0.9, 0.6, 0.2, 0.1]
overlap    = 0.9·1 + 0.6·1 + 0.2·0 + 0.1·0 = 1.5
sum of p   = 1.8        sum of y = 2
Dice       = 2·1.5 / (1.8 + 2) = 3 / 3.8 = 0.789
Dice loss  = 1 − 0.789 = 0.211
```

BCE for the same four pixels: `(0.105 + 0.511 + 0.223 + 0.105) / 4 = 0.236`.

**Our loss = BCE + Dice loss = 0.236 + 0.211 = 0.447.**

Why both? Only a small share of the pixels are vessels. BCE counts every pixel equally, so the huge background dominates. Dice ignores correctly predicted background, so it pushes the network to really find vessels. BCE gives smooth, stable gradients. Together they work better than alone.

Code: `bce_dice_loss` in `src/medskel_ai/train.py`.

## 8. Backpropagation and Adam

**Gradient**: for each weight, the gradient says "if I increase this weight a tiny bit, how much does the loss change?".

**Backpropagation** computes all gradients (7.8 million of them) in one backward sweep using the chain rule. PyTorch does it with `loss.backward()`.

One-weight example. Model: `z = w·x`. Take `x = 2`, true label `y = 1`, current weight `w = 0.5`.

1. Forward: `z = 0.5·2 = 1`. `p = sigmoid(1) = 0.731`. Loss `= −ln(0.731) = 0.313`.
2. Backward: for BCE with sigmoid, `dLoss/dz = p − y = 0.731 − 1 = −0.269`. Chain rule: `dLoss/dw = (p − y)·x = −0.269·2 = −0.538`.
3. The gradient is negative. So **increasing** `w` lowers the loss.
4. Plain gradient descent with learning rate 0.1: `w = 0.5 − 0.1·(−0.538) = 0.554`.
5. Check: new `z = 1.108`, `p = 0.752`, loss `= 0.285`. Lower than 0.313. It learned.

**Adam** is a smarter version of step 4. For each weight it remembers:

- the average of recent gradients (which direction, like momentum), and
- the average of recent squared gradients (how big they usually are).

The step is `learning rate × average gradient / √(average squared gradient)`. So every weight gets its own step size. Weights with tiny gradients still move. Weights with huge gradients do not explode. We start with learning rate 0.001 and lower it smoothly to 0 during training (cosine schedule).

**Mixed precision** is a speed trick: most computations use 16-bit numbers instead of 32-bit. It halves the memory. It is why the model fits on a 4 GB GPU.

One training step in code (`src/medskel_ai/train.py`):

```python
logits = model(images)                 # forward
loss = bce_dice_loss(logits, masks)    # how wrong?
loss.backward()                        # backpropagation
optimizer.step()                       # Adam update
```

## 9. The metrics

Every pixel falls in one of four boxes:

| | expert says vessel | expert says background |
|---|---|---|
| **we say vessel** | TP (found) | FP (false alarm) |
| **we say background** | FN (missed) | TN (correctly ignored) |

Example with 10 pixels: `TP = 3, FP = 1, FN = 1, TN = 5`.

| Metric | Formula | Example | Question it answers |
|---|---|---|---|
| Dice (= F1) | 2TP / (2TP + FP + FN) | 6/8 = 0.75 | How well do the two masks overlap? |
| IoU | TP / (TP + FP + FN) | 3/5 = 0.60 | Same idea, stricter |
| Precision | TP / (TP + FP) | 3/4 = 0.75 | Of the pixels we marked, how many are real vessels? |
| Recall (sensitivity) | TP / (TP + FN) | 3/4 = 0.75 | Of the real vessel pixels, how many did we find? |
| Specificity | TN / (TN + FP) | 5/6 = 0.83 | Of the background pixels, how many did we leave alone? |
| Accuracy | (TP + TN) / all | 8/10 = 0.80 | How many pixels are correct in total? |

### Why accuracy is misleading

Vessels are a small minority of the pixels. Imagine a lazy model that paints **everything black**. If 8% of the pixels are vessels, this model is 92% "accurate". But it found zero vessels: recall 0, Dice 0. Accuracy mostly measures the easy background. Specificity has the same weakness. That is why we judge the model by **Dice, IoU, precision and recall**. Our real "all black" baseline number is in section 10.

## 10. Our real results

All numbers below come from one real run on this laptop (RTX 3050, 4 GB) on 14 November 2026. The raw files are in `results/`.

**Data.** FIVES has 600 official training images and 200 official test images, each 2048 x 2048. We removed 2 training images whose masks were completely black (label errors). From the rest we took 88 images for validation (22 per disease group) and trained on 510. In the training masks, vessels are only **7.5%** of the pixels.

**Training.** 38 epochs in 51 minutes. Each step used 4 random 512 x 512 crops. The training loss fell from 1.00 to 0.16. The validation loss fell from 0.83 to 0.21. The best epoch was number 32 (validation Dice 0.878 on fixed crops). We kept the weights of that epoch.

**Threshold.** On the 88 whole validation images we tried thresholds from 0.10 to 0.90. The best mean Dice was at **0.40** (Dice 0.899). We fixed 0.40 **before** opening the test set.

**Test.** We evaluated the 200 official test images **one time**:

| Metric | Mean over 200 images | Spread (std) |
|---|---|---|
| Dice | **0.875** | 0.114 |
| IoU | 0.791 | 0.136 |
| Precision | 0.909 | 0.063 |
| Recall | 0.856 | 0.144 |
| Specificity | 0.993 | 0.005 |
| Accuracy | 0.986 | 0.006 |

The median Dice is 0.907. So the typical image is better than the mean. A few very bad images pull the mean down.

Per disease group (50 test images each):

| Group | Dice | Precision | Recall |
|---|---|---|---|
| A (AMD) | 0.908 | 0.925 | 0.898 |
| D (diabetic retinopathy) | 0.876 | 0.925 | 0.840 |
| G (glaucoma) | 0.828 | 0.901 | 0.794 |
| N (normal) | 0.887 | 0.886 | 0.892 |

What to notice:

- **Precision (0.91) is higher than recall (0.86).** When the model marks a vessel it is usually right. But it misses about 14% of the vessel pixels. These are most likely the thin, faint ones (we did not measure thin vessels separately).
- **Glaucoma is the weakest group**, with the largest spread. The two worst images are `122_G` (Dice 0.03) and `123_G` (Dice 0.11). There the expert marked very few vessels (0.7% and 1.7% of the pixels) and the model found almost nothing. 10 of the 200 images have Dice below 0.7.
- **The accuracy trap, with our real numbers.** A model that paints everything black gets **92.9%** accuracy on the test set, with Dice 0 and recall 0. Our model gets 98.6%. Both numbers look high. Only Dice (0 against 0.875) shows the real difference.

## 11. Limits — say these out loud

1. **Patients may overlap.** FIVES has 800 images from 573 patients, and no patient ids are published. So images of the same person can be in training and in validation, or in training and in test. Our scores may be a little optimistic. We cannot claim "tested on new patients".
2. **Duplicate photos exist.** We found 6 cases where the same photo is stored under two names (5 in train, 1 in test, none shared between train and test). We keep such copies in the same part of the split.
3. **One run, one seed.** We did not repeat the training. We do not know how much the result changes with another seed.
4. **One dataset, one camera type.** We did not test on images from another hospital or camera. Results there will probably be worse.
5. **The expert masks are not perfect truth.** Two masks were empty. Some may be misaligned. Experts also disagree with each other.
6. **No field-of-view mask.** The black corners around the retina count as easy background pixels. This pushes accuracy and specificity up. It does not change Dice much.
7. **The validation score during training used crops from the image centre**, not whole images. It was only used to pick the best epoch.
8. **Pixels only.** The model gives a mask. It does not measure vessel width or length. It does not diagnose anything. It is not a medical device.

## 12. Ten likely professor questions

**1. Why a U-Net and not a normal classification CNN?**
A classification CNN gives one answer for the whole image. We need one answer per pixel, at the same size as the input. The U-Net has a decoder that brings the picture back to full size, and skip connections that bring back sharp detail.

**2. What do the skip connections do? What happens without them?**
They copy encoder feature maps to the decoder and concatenate them. Without them, the decoder only has the small 32 x 32 map (for a 512 crop). All fine position information was lost by four poolings. Thin vessels would come out blurry or disappear.

**3. Why do you train on 512 x 512 crops and not on the whole image, or on a resized image?**
A whole 2048 x 2048 image does not fit in 4 GB of GPU memory during training. Resizing to a small image would erase vessels that are 1–2 pixels wide. Crops keep the full resolution and fit in memory. Random crops also give variety: every epoch sees new windows.

**4. How do you predict a whole image then?**
We cut it into 512 x 512 tiles that overlap by 64 pixels. We predict each tile. We average the probabilities where tiles overlap. This works because convolutions do the same thing at every position.

**5. Why BCE plus Dice? Why not only one?**
Only 7.5% of the pixels are vessels. BCE treats all pixels equally, so background dominates. Dice measures overlap of the vessel class and ignores true negatives. BCE gives smooth gradients for every pixel. The sum is a common, stable choice. We did not run a comparison of the two, so we cannot claim it is better on our data.

**6. Why is the output a logit and not a probability?**
`BCEWithLogitsLoss` combines sigmoid and logarithm in one stable formula. If we computed sigmoid first and then `log`, a probability very close to 0 would give `log(0)` = minus infinity. For prediction we apply sigmoid ourselves.

**7. Why is accuracy a bad metric here?**
Classes are very unbalanced. An all-black prediction gets 92.9% accuracy on our test set with zero vessels found. Dice, IoU, precision and recall look at the vessel class and expose this.

**8. How did you choose the threshold 0.40? Is that cheating?**
We chose it on the 88 validation images, by trying 17 values and taking the best mean Dice. The test images were not used. The curve is flat: any value from 0.30 to 0.45 gives Dice within 0.001. Choosing it on the test set would be cheating, because the test score would then be tuned to the test data.

**9. Is the model overfitting? How do you know?**
At the end the training loss is 0.16 and the validation loss is 0.21. The gap is small, and the validation loss did not rise again. So there is no strong overfitting. Things that help: random crops, flips, rotations, brightness changes, and keeping the best epoch by validation Dice. The test Dice (0.875) is lower than the validation Dice (0.899), which is a normal small drop.

**10. How many parameters, and where are most of them?**
7,849,601. A 3 x 3 convolution has `9 × input channels × output channels` weights. So the layers with many channels are the big ones: the bottleneck (256→512→512) alone has about 3.5 million, and the first decoder block about 2.4 million. The first layer has only 864 weights.

**Bonus: what would you do next?**
Train several seeds to get error bars. Test on another dataset (CHASE_DB1). Measure thin vessels separately. Look by eye at the worst images to see whether the model or the label is wrong.
