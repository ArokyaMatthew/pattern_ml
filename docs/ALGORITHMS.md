# ID3 and C4.5: theory and implementation notes

This document explains the two algorithms as implemented in `arbor/`, with
references to the code. For how the modules fit together, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## 1. The common skeleton

Both algorithms grow a tree top-down and greedily (Hunt's method):

```
grow(S, depth):
    node ← leaf predicting the majority class of S
    if stopping rule holds for S: return node
    for each candidate attribute A:
        score(A)                         # gain for ID3, gain ratio for C4.5
    A* ← best admissible candidate
    if none: return node
    partition S by the outcomes of A*
    for each outcome v: node.child[v] ← grow(S_v, depth + 1)
    return node
```

`BaseDecisionTree._grow` in `arbor/tree.py` is this loop. The subclasses supply
`_should_stop`, `_score`, `_select` and, for C4.5, `_post_grow` (pruning).

Every case carries a **weight**. For ID3 the weights are always 1. C4.5 needs
fractional weights to handle unknown values (section 4.4), so all counting
uses `np.bincount(..., weights=w)`; with unit weights the result is plain
counts.

## 2. Measuring impurity

All functions live in `arbor/criteria.py`.

**Entropy** of a set *S* with class proportions *p₁ … p_k*:

$$H(S) = -\sum_{i=1}^{k} p_i \log_2 p_i \qquad (0 \log_2 0 = 0)$$

It is 0 for a pure set and log₂ *k* bits when all *k* classes are equally
frequent.

**Information gain** of splitting *S* on attribute *A*:

$$\operatorname{Gain}(S,A) = H(S) - \sum_{v} \frac{|S_v|}{|S|}\, H(S_v)$$

**Split information**, the entropy of the partition sizes themselves:

$$\operatorname{SplitInfo}(S,A) = -\sum_{v} \frac{|S_v|}{|S|}\log_2\frac{|S_v|}{|S|}$$

**Gain ratio**:

$$\operatorname{GainRatio}(S,A) = \frac{\operatorname{Gain}(S,A)}{\operatorname{SplitInfo}(S,A)}$$

### Worked example: Play Tennis

The 14 days contain 9 *yes* and 5 *no*, so *H(S)* = 0.940 bits. Splitting on
*outlook* gives sunny (2 yes, 3 no), overcast (4, 0) and rain (3, 2):

| | sunny | overcast | rain |
|---|---|---|---|
| cases | 5 | 4 | 5 |
| entropy | 0.971 | 0 | 0.971 |

Gain = 0.940 − (5/14·0.971 + 4/14·0 + 5/14·0.971) = **0.2467**.
SplitInfo = H(5/14, 4/14, 5/14) = 1.5774, so GainRatio = **0.1564**.
The other attributes score Gain(humidity) = 0.1518, Gain(wind) = 0.0481 and
Gain(temperature) = 0.0292, so both algorithms put *outlook* at the root. The
app's *Split analysis* table shows exactly these numbers, and
`tests/test_id3.py` checks them.

## 3. ID3

`arbor/id3.py`, after Quinlan (1986).

| Aspect | Implementation |
|---|---|
| Attribute types | Categorical only. `_feature_kind` reports every column as categorical; numeric columns must be discretised first (section 5) |
| Split | One branch per value present at the node |
| Criterion | Highest information gain; ties go to the earlier column |
| Re-use | An attribute is never tested twice on one path (`used` set in `_grow`) |
| Stopping | Pure node; no attributes left; fewer than `min_samples_split` cases; `max_depth` reached; best gain ≤ `min_gain` |
| Missing values | Turned into a category `"(missing)"`. In the app they are imputed before ID3 sees them |
| Unseen value at prediction | The case stops at that node and takes the node's class distribution |

ID3's weakness is visible in its criterion. Gain favours attributes with many
values: a unique ID column splits every case into its own pure leaf and gets
the maximum possible gain while predicting nothing. On real data the same bias
produces bushy, overfitted trees, as the depth sweep in the app shows.

## 4. C4.5

`arbor/c45.py`, after Quinlan (1993), following Release 8 where the two differ.

### 4.1 Gain ratio with the average-gain rule

Gain ratio fixes ID3's bias, but it has a failure mode of its own: an attribute
whose split information is almost zero (one dominant branch) gets a huge ratio
from a negligible gain. C4.5 therefore:

1. computes the gain of every candidate,
2. admits only those whose gain is at least the **average** gain of the candidates with positive gain,
3. picks the admitted candidate with the highest gain ratio.

`C45Classifier._select` implements this; rejected candidates are labelled
*below average gain* in the split analysis table.

### 4.2 Continuous attributes

For a numeric attribute the cases at the node are sorted once. Every boundary
between two *distinct* adjacent values is a candidate threshold, and the
class counts on each side of every boundary come from a single cumulative sum
(`_score_continuous`). The whole search is vectorised: *O(n log n)* for the
sort and *O(n·k)* for the scan, with *k* classes.

* The threshold is the midpoint of the two adjacent values.
* The cut is chosen by **gain**, then Release 8's correction is subtracted:
  log₂(number of candidate cuts) / |S| bits. This penalises attributes that
  win only because they offered many cuts to choose from. The correction can
  make a gain negative; the app then shows that attribute as not admissible.
* A continuous attribute can be tested again further down with a different
  threshold. Categorical attributes are used once per path, as in ID3.

### 4.3 Minimum cases

A split is admissible only if at least two of its branches receive at least
`min_samples_leaf` cases (C4.5's `-m` option, default 2). A node with fewer
than 2·*m* cases becomes a leaf.

### 4.4 Unknown values

When *C4.5 native* is selected in the app, missing values reach C4.5
untouched.

**Scoring.** Let *F* be the fraction of the node's weight whose value of *A* is
known. Gain is computed on the known cases only and scaled:

$$\operatorname{Gain}(S,A) = F \cdot \big(H(S_{known}) - \sum_v \tfrac{|S_v|}{|S_{known}|} H(S_v)\big)$$

and split information counts *unknown* as one more outcome, which lowers the
gain ratio of attributes with many gaps.

**Partitioning.** A case with an unknown value is sent down **every** branch,
with its weight multiplied by that branch's share of the known weight
(`BaseDecisionTree._partition`). The weights reaching the children still add up
to the parent's weight, so leaves can hold fractional counts such as
`yes (3.38/0)`.

**Prediction.** A case whose value is unknown at a node is routed down every
branch with the same proportions, and the leaf class distributions it reaches
are averaged by those weights (`BaseDecisionTree._route`). A categorical value
never seen in training is treated the same way.

### 4.5 Error-based pruning

After the full tree is grown, C4.5 asks bottom-up whether each subtree would
make fewer errors on unseen cases than a single leaf would.

A leaf that covers *N* training cases and misclassifies *E* of them is assumed to
have a true error rate equal to the upper limit *U_CF(E, N)* of a binomial
confidence interval at level CF. Its predicted errors are *N · U_CF(E, N)*. For
*E* ≥ 1 the normal approximation with *z* = Φ⁻¹(1 − CF) is used:

$$U_{CF}(E,N) = \frac{E + \tfrac12 + \tfrac{z^2}{2} + z\sqrt{(E+\tfrac12)\big(1-\tfrac{E+1/2}{N}\big) + \tfrac{z^2}{4}}}{N + z^2}$$

and for *E* < 1 the exact binomial expressions. `added_errors` in
`arbor/c45.py` is a line-by-line port of C4.5's `AddErrs` routine. The test
suite checks it against Quinlan's published values: *U*₂₅%(0, 6) = 0.206,
*U*₂₅%(0, 9) = 0.143 and *U*₂₅%(1, 16) = 0.157.

`C45Classifier._prune` walks the tree bottom-up. For each internal node it
compares

* the estimated errors if the node became a leaf, with
* the sum of estimated errors of its (already pruned) leaves,

and replaces the subtree with a leaf when the first is no more than the second
plus 0.1. Lower CF gives larger error estimates and more pruning. The app's
split analysis shows both estimates for every node that was considered.

C4.5 also performs *subtree raising* (replacing a node by its largest branch).
That is not implemented here.

## 5. Discretisation for ID3

`TreePreprocessor` in `arbor/preprocessing.py` cuts every numeric attribute into
at most *n* intervals, learning the cut points on the training split:

| Strategy | Cut points |
|---|---|
| Equal frequency (`quantile`) | Training quantiles, so each interval holds roughly the same number of cases |
| Equal width (`uniform`) | Evenly spaced between the training minimum and maximum |
| k-means (`kmeans`) | Boundaries between 1-D k-means clusters (scikit-learn's `KBinsDiscretizer`) |

If an attribute has no more distinct values than the requested number of
intervals, each value gets its own interval. Intervals are right-closed and
the outer ones are open-ended (`≤ 5.1`, `(5.1, 5.8]`, …, `> 6.4`), so test values
outside the training range still fall into a bin. Equal frequency is the
default because it adapts to skewed distributions.

## 6. Evaluation

`arbor/metrics.py` wraps `sklearn.metrics`. For each class *c*, from the
confusion matrix:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}\qquad \text{Recall}_c = \frac{TP_c}{TP_c+FN_c}\qquad F_{1,c} = \frac{2\,P_c R_c}{P_c + R_c}$$

* **Macro** average: unweighted mean over classes, so every class counts equally (the default, and the honest choice for imbalanced data).
* **Weighted** average: mean weighted by class support.
* **Accuracy**: share of all cases classified correctly.

A class that is never predicted gets precision 0 rather than an error
(`zero_division=0`).

**Cross-validation** (`experiment.cross_validate_models`) uses stratified *k*-fold
when every class has at least *k* cases and plain shuffled *k*-fold otherwise.
Preprocessing is part of each algorithm's pipeline, so imputation values and
bin edges are refitted inside every fold.

## 7. Feature importance

The importance of an attribute is the total gain of the splits that use it,
each weighted by the share of training cases reaching that node, normalised to
sum to 1 (`BaseDecisionTree._finalize`). This is the information-gain version
of scikit-learn's impurity-based importance.

## References

* J. R. Quinlan. *Induction of decision trees.* Machine Learning 1(1), 81–106, 1986.
* J. R. Quinlan. *C4.5: Programs for Machine Learning.* Morgan Kaufmann, 1993.
* J. R. Quinlan. *Improved use of continuous attributes in C4.5.* JAIR 4, 77–90, 1996.
* T. M. Mitchell. *Machine Learning*, chapter 3. McGraw-Hill, 1997.
