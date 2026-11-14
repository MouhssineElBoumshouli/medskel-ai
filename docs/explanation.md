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

RESULTS_PLACEHOLDER

## 11. Limits — say these out loud

LIMITS_PLACEHOLDER

## 12. Ten likely professor questions

QA_PLACEHOLDER
