# 05 — Skeletonization and retinal vessel measurements: definitions and limits

## Essential distinction

1. **Segmentation** answers: which image pixels depict blood vessels?
2. **Skeletonization / graph extraction** answers: where are the approximate centerlines, branches and junctions inside that mask?
3. **Morphometry** answers: what are lengths, widths, densities and twisting measures under a specified protocol?
4. **Measurement reproducibility** asks: do people, tools or repeat images yield similar numbers?
5. **Clinical validity** asks: do these quantities reliably correspond to a clinical target under the defined use, population, calibration and intended purpose?

One strong result at level 1 **does not prove** levels 2–5. An association between vascular features and blood pressure or glycated hemoglobin does not give a valid individual blood pressure reading or diabetes diagnosis.

## Presently existing MedSkel code and parameters

The old pipeline is `binary mask → boundary polygon → simplify → densely sample boundary → Voronoi edges / medial-axis filtering → graph → branch measurements`. `epsilon` controls polygon simplification tolerance in pixels; `theta_deg` filters low-separation-angle Voronoi hairs (README default example 70°); `prune` removes short leaf branches relative to local radius. High simplification may stabilize global measures while losing true detail and worsening positional agreement.

Classic **pixel thinning** repeatedly reduces the mask to a one-pixel-wide line and is a standard fast baseline. It is not the same data representation as MedSkel's subpixel polygon/Voronoi graph; fair comparison requires common branch-extraction and measurement rules.

### Geometry caution

The wavefront construction in `medskel/bisector.py` yields a *straight skeleton* for implemented cases, and halts at unhandled split events. The medial axis defined by maximal inscribed disks can differ at concave/reflex corners. The primary Voronoi method approximates the medial axis of a sampled polygon boundary. Do not conflate these objects in a scientific presentation.

## Measurement definitions and units

| Measurement | Plain description | Candidate computation | Important caution |
|---|---|---|---|
| Total vessel length | How long all extracted centerlines are | Sum of consecutive centerline segment Euclidean lengths, in **pixels** | A higher number can mean extra spurs/false vessels; a lower number can mean missing small vessels; optical scale unknown |
| Branch count | Number of centerline segments between junctions/endpoints | Reduce skeleton graph to branch edges | Sensitive to connectivity, diagonal adjacency, false branches and pruning |
| Bifurcation/junction count | Number of places where one line splits | Count nodes with degree≥3 under an explicitly stated graph definition | Crossing vessels and adjacent pixel clusters can produce errors; classic thinning needs care |
| Approximate vessel calibre | Width of vessel segment | Local `2 × distance_transform(mask)` along centerline, aggregate per branch | Pixel units, biased by segmentation, interpolation and image scale; not CRAE or CRVE |
| Tortuosity | How curvy a branch is | `arc length / straight-line chord length` for a defined branch | Undefined if chord≈0; branching and graph simplification change segment definition; other tortuosity definitions exist |
| Centerline position agreement | Whether two skeletons occupy the same locations | Fraction within specified pixel tolerance of paired skeleton | Agreement does not establish ground-truth anatomical centerline |
| Coverage | Does estimated skeleton reach the expected/reference vessel extent? | Fraction of reference points with a nearby predicted skeleton point | Prevents degenerate method that returns almost nothing from appearing artificially precise |
| Vessel density | Fraction of an image region covered by vessels | Vessel-mask area / specified valid FOV/ROI area | Needs precise region, QC and resolution handling |
| CRAE/CRVE/AVR | Standardized central artery/vein caliber summaries | Specific vessel selection + optic disc region + accepted formula; AVR=CRAE/CRVE | **Not implemented/validated**; requires artery-vs-vein labels and calibrated research protocol |
| Fractal dimension, branching angle | Network complexity and branch geometry | Defined box counting / tangent-angle methods | Protocol-sensitive; **not a current validated product feature** |

### Example of width math

Inside a binary vessel mask, `distance_transform_edt` gives each foreground pixel's distance to its nearest background pixel. For a centerline point at pixel `(x,y)`, the local radius estimate is `r = DT[y,x]`; approximate diameter is `2r` pixels. This is a local geometric approximation, affected by mask thickness and discretization, not an instrument-measured micrometre dimension.

### Example of length math

For a polyline with points `P_0 ... P_k`, length is `L = Σ_{i=0}^{k-1} sqrt((x_{i+1}-x_i)^2+(y_{i+1}-y_i)^2)`. The absolute number depends on pixel spacing and image scale. Older pixel-thinning paths can have a staircase-related absolute-length bias; cross-method absolute comparison should account for this rather than assuming differences solely come from morphology.

### Simplification vs accuracy failure

Suppose an algorithm removes many thin side branches in *both* human observer masks. Its total length might change very little between observers, yielding an apparently excellent **reproducibility** score. But it may be inaccurate or incomplete for vessel anatomy. Therefore always report signed length bias/absolute differences, retained skeleton coverage, small-vessel recall and position. Avoid declaring MedSkel 'superior' from one stability metric alone.

## What can be claimed now

- [VERIFIED] Original code computes a family of pixel-space vessel graph measurements from masks.
- [REPORTED] On CHASE two-observer masks, original MedSkel has stronger length stability (3.2% vs 9.3%), but worse positional agreement (71.6% vs 80.9%) than thinning at reported settings; runtime is worse in the README's one-retina example.
- [NOT PROVEN] That either method matches the real biological vessel centerline more accurately on heterogeneous retinas.
- [NOT IMPLEMENTED] Standardized calibrated CRAE/CRVE/AVR, artery/vein classification, risk scores or clinical predictions.
- [PLANNED] Feed learned segmentation to geometry and measure whether it changes performance in a fair controlled comparison.

## Test ideas for geometry integration

- Input an empty mask, an isolated line, a Y junction, two disconnected tubes, a loop, a thin diagonal branch and a cross; compare branch/width/length expectations with hand-derived truth.
- Test coordinate x/y order and preserved resize/scale metadata.
- Compare MedSkel and thinning on **identical** expert masks at multiple epsilon/prune settings.
- Check that radius and centerline positions remain inside the mask as expected; quantify loss of components and tiny branches.
- Measure runtime with same image resolution and hardware.
- Review vessels near optic disc, crossings, faint thin vessels, holes, noisy boundaries and low contrast.
- Treat two-observer CHASE as a study of interobserver variability, not repeat photography or proven anatomical gold standard.
