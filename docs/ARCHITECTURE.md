# Architecture

The repository has two layers with a one-way dependency:

```
ui/   (Streamlit: layout, widgets, caching)  ──depends on──▶  arbor/   (pure Python library)
```

`arbor/` never imports Streamlit. Everything in it can be used from a script,
a notebook or the tests, and the UI is a thin shell around it.

## The library: `arbor/`

| Module | Responsibility |
|---|---|
| `criteria.py` | Entropy, conditional entropy, information gain, split information, gain ratio. Works on weighted counts |
| `tree.py` | `Node`, `SplitCandidate`, `Rule`, and `BaseDecisionTree`: input encoding, weighted recursive growth, partitioning (including fractional cases), prediction by routing weights, and inspection (`rules`, `export_text`, `decision_path`, `feature_importances_`) |
| `id3.py` | `ID3Classifier`: categorical-only scoring by information gain, ID3's stopping rules |
| `c45.py` | `C45Classifier`: gain ratio with the average-gain rule, vectorised threshold search, unknown-value handling, minimum cases, error-based pruning (`added_errors`) |
| `preprocessing.py` | CSV reading and tidying, column profiling, target and identifier detection, discretisation, and `TreePreprocessor` (a scikit-learn transformer) |
| `experiment.py` | `ExperimentConfig` (frozen, hashable), pipeline construction per algorithm, hold-out runs, cross-validation, depth sweep |
| `metrics.py` | Thin wrappers over `sklearn.metrics` |
| `graph.py` | Builds styled `graphviz.Digraph` objects from fitted trees |
| `charts.py` | Matplotlib figures |
| `style.py` | The colour tokens shared by charts, tree drawings and the stylesheet |
| `datasets.py` | Registry of the bundled CSV files |

### Estimators follow scikit-learn's contract

`ID3Classifier` and `C45Classifier` derive from `ClassifierMixin` and
`BaseEstimator`. Constructor arguments are stored unchanged, learned state ends
with an underscore (`root_`, `classes_`, `feature_names_in_`…), and
`fit`/`predict`/`predict_proba`/`score` behave as scikit-learn expects. As a
result `clone`, `Pipeline`, `cross_validate` and `GridSearchCV` all work
without adapters.

### One pipeline per algorithm

```
ID3   = Pipeline([prep: TreePreprocessor(impute=True,  discretize=True),  tree: ID3Classifier])
C4.5  = Pipeline([prep: TreePreprocessor(impute=depends, discretize=False), tree: C45Classifier])
```

The two algorithms need different inputs, so each gets its own preprocessing.
Putting preprocessing inside the pipeline is what prevents leakage: the
pipeline is fitted on the training split only, and cross-validation refits it
from scratch in every fold.

### Data representation inside a tree

During `fit` every column is encoded once:

* categorical → integer codes (−1 = missing) plus the list of categories;
* continuous → `float64` with `NaN` for missing.

Growth works on index arrays and a parallel array of case weights. A node's
class counts are one `np.bincount(y[idx], weights=w)`. The contingency table of
a categorical attribute is one `bincount` over `code · k + class`. The
threshold search for a continuous attribute is one sort plus one cumulative
sum. Nothing loops over cases in Python.

At prediction time `_route` sends **arrays** of row indices with weights down
the tree, splitting them by mask at each node, so a test set is classified in
one pass per node rather than one walk per row. Unknown values multiply their
weight by the branch shares (C4.5) or stop at the node (ID3).

## The interface: `ui/` and `app.py`

| Module | Responsibility |
|---|---|
| `app.py` | Page setup, the overall flow (dataset → target → schema → config → experiment → tabs) |
| `ui/sidebar.py` | All setup controls; returns `ExperimentConfig` keyword arguments |
| `ui/cache.py` | `st.cache_data` wrappers around loading, training, cross-validation and the depth sweep |
| `ui/views/*.py` | One module per tab |
| `ui/components.py` | HTML building blocks (header, stat strip, prediction card) and figure display |
| `ui/theme.py` | The stylesheet |
| `.streamlit/config.toml` | Theme (colours, IBM Plex fonts) and server settings |

### Responsiveness

Streamlit re-runs the script on every interaction, so two measures keep the app fast:

1. **Caching on the configuration.** `ExperimentConfig` is a frozen dataclass, so it can be part of a cache key. Changing a setting and changing it back costs nothing; only genuinely new settings train new trees.
2. **Fragments.** Controls that only affect one view (tree layout, levels shown, averaging mode, the prediction form, cross-validation folds) live inside `st.fragment` functions, so they redraw their own section and not the whole page.

### Visual design

* One accent colour (deep green) for interactive elements; all other colour carries data.
* Algorithm identity: ID3 blue, C4.5 orange, everywhere. Class identity: a fixed eight-colour categorical palette, assigned in the same order in tree nodes, class-balance bars and probability bars. The palette was validated for colour-vision-deficiency separation; every coloured mark is also labelled, so colour never carries meaning alone.
* Confusion matrices use a single-hue sequential ramp; text on cells switches between white and ink by luminance.
* Typography: IBM Plex Sans and IBM Plex Mono, tabular figures in tables.

## Tests

`tests/` uses pytest and runs in a few seconds:

| File | Covers |
|---|---|
| `test_criteria.py` | Entropy, gain, split information and gain ratio against textbook values |
| `test_id3.py` | Mitchell's Play Tennis tree, attribute re-use, limits, unseen and missing values, rules, importances, scikit-learn compatibility |
| `test_c45.py` | Quinlan's golf tree, thresholds, the average-gain rule, the cut-point penalty, fractional cases, blended prediction, `added_errors` vs Quinlan's tables, pruning monotonicity, minimum cases |
| `test_preprocessing.py` | CSV parsing, type inference, target/ID detection, binning, leakage-free imputation and discretisation |
| `test_experiment.py` | Hold-out runs, reproducibility, per-algorithm inputs, cross-validation, depth sweep, CART reference, metrics, tree drawing, charts |
