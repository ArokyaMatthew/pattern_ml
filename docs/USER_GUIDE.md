# User guide

A tour of the workbench, tab by tab, followed by experiments that show the
differences between ID3 and C4.5. Start the app with `streamlit run app.py`
(see the README for setup).

## The layout

* **Sidebar** – the experiment setup, numbered in the order you would normally work through it.
* **Header** – the dataset, its size, the target and the number of classes.
* **Tabs** – results. Every sidebar change retrains both trees and updates every tab.

Settings you change stay in place when you switch tabs. *Reset settings* at
the bottom of the sidebar restores the defaults.

## Sidebar

### 01 Dataset

Choose a bundled sample or **Upload CSV**. The samples:

| Sample | Rows | Why it is there |
|---|---:|---|
| Iris (default) | 150 | Clean numeric data, three classes; small, readable trees |
| Play Tennis | 14 | The textbook ID3 example; all attributes categorical |
| Weather (numeric) | 14 | The same days with numeric temperature and humidity; shows C4.5's thresholds |
| Wine | 178 | 13 numeric attributes, three classes |
| Breast Cancer (Wisconsin) | 569 | 30 numeric attributes, two classes |
| Credit Risk (synthetic) | 600 | Mixed numeric and categorical attributes with missing values; the best showcase for C4.5 |

Datasets under 30 rows switch the hold-out test set off when they load: they
are meant to be learned whole, and 4 or 5 test cases say nothing reliable.

### 02 Target

The column to predict. Arbor suggests one (a column named `class`, `label`,
`species`… or the last column). If you choose an unsuitable column, such as a
continuous number or an identifier, the app explains why and waits for another
choice.

### 03 Preprocessing

| Control | Effect |
|---|---|
| Missing values | **Impute** fills gaps with the training median (or mean) and mode. **Drop** removes incomplete rows. **C4.5 native** passes gaps to C4.5 untouched (fractional cases) while ID3 still gets imputed data |
| Numeric fill value | Median or mean, for imputation |
| ID3 intervals per numeric attribute | Upper bound on the intervals each numeric column is cut into for ID3 |
| ID3 binning strategy | Equal frequency, equal width or k-means |
| Remove duplicate rows | Drops exact duplicates before splitting |

### 04 Validation

| Control | Effect |
|---|---|
| Hold out a test set | On: train on one part, score on the rest. Off: train and score on everything (optimistic) |
| Test set size | 10–50 % |
| Stratify by class | Keep class proportions equal in both parts |
| Random seed | Change it to see how much the results depend on the particular split |

### 05 ID3 and 06 C4.5

| Control | Algorithm | Effect |
|---|---|---|
| Max depth | both | 0 means unlimited |
| Min cases to split a node | ID3 | Nodes with fewer cases become leaves |
| Min information gain | ID3 | A split must gain more than this many bits |
| Min cases per branch (*m*) | C4.5 | A split must send at least *m* cases down two of its branches |
| Error-based pruning | C4.5 | Prune after growing |
| Confidence factor (CF) | C4.5 | Lower values prune harder |

## Tabs

### Data

* **Summary strip** – rows, attributes in use, classes, missing cells, and the *majority baseline*: the accuracy of always predicting the most common class. A model has to beat this to be useful.
* **Attributes** – the schema editor. Untick a column to exclude it; switch *Treat as* between Numeric and Categorical. A numeric column of codes (1, 2, 3 meaning categories) should be categorical.
* **Class balance** – how often each class occurs. A warning appears when one class is three or more times as large as another.
* **Preprocessing** – the exact steps each algorithm receives, including the learned fill values and ID3's cut points.
* **Rows** – the raw file, or the training rows *as ID3 sees them* (intervals) and *as C4.5 sees them*.

### Trees

* Pick **ID3** or **C4.5** and a layout. **Levels shown** limits how deep the drawing goes (deeper subtrees collapse to a dashed *+N nodes* stub); the full-screen button on the drawing helps with big trees.
* Each node shows its attribute or predicted class, the number of training cases *n*, its entropy *H*, the gain (ID3) or gain ratio (C4.5) of its split, and a bar with the class mix. Leaves are tinted with their predicted class.
* **Why each split** – choose any node to see every candidate attribute with its gain, split information, gain ratio, cut point (C4.5) and status: *chosen*, *below average gain*, *one value at this node*, and so on. For C4.5 it also shows the pruning check (estimated errors as a leaf vs as a subtree).
* **The tree as rules** – one IF-THEN rule per leaf with its support and confidence, or the Weka/J48-style text tree.
* Downloads: DOT source (open it in any Graphviz viewer), rules as text, and SVG when Graphviz is installed on the system.

### Evaluation

For each algorithm: accuracy (with the count of correct cases), precision,
recall and F1-score, a confusion matrix (rows are actual classes, columns
predicted; shading is the share of each actual class), and per-class scores.
Switch the averaging between **Macro** (each class counts equally) and
**Weighted** (by support).

### Comparison

* **Head to head** – a sentence summarising who wins and by how much, a metric chart, and a scorecard including training accuracy, the train–test gap (a direct measure of overfitting), tree size and timing.
* **What the trees rely on** – feature importance for both algorithms.
* **k-fold cross-validation** – 3–10 folds over the whole dataset, preprocessing refitted per fold. The dot plot shows every fold, the bar marks the mean. Much steadier than a single split on small datasets.
* **Depth limit sweep** – both trees retrained at depth 1, 2, 3… Training accuracy keeps rising while test accuracy stalls or falls: that gap is overfitting.
* **Reference point** – optionally adds scikit-learn's CART as a sanity check.

### Predict

* **Classify a new case** – edit any value and both trees answer at once with class probabilities and the path taken (for ID3 on numeric attributes, the interval the value fell into). *Fill the form from* loads a test row so you can check a known answer. Choose *(unknown)* or clear a number to see missing-value handling.
* **Show the decision path in the tree** – the tree with the case's path outlined; in large trees only the path is expanded.
* **Test-set predictions** – every test case with the actual class and both predictions. Wrong answers are shaded; filter to disagreements or to one algorithm's errors; download as CSV.
* **Score a file** – upload new cases (same attribute columns, target optional) and download both models' predictions with their confidence.

### Method

The theory behind both algorithms, with formulas, in the order the code applies it.

## Experiments worth trying

1. **The textbook tree.** Play Tennis, Trees tab, ID3. Compare the root's split analysis with the gains in Mitchell's book: outlook 0.247, humidity 0.152, wind 0.048, temperature 0.029.
2. **Why ID3 needs help.** Upload a CSV with a unique ID column (for example Credit Risk with a `customer_id` column added) and tick the ID back on in the schema. ID3 splits on it at the root, because it has the maximum possible gain, and grows one useless leaf per row. C4.5 refuses it: no branch holds the minimum of *m* = 2 cases, and its split information of log₂ *n* bits drags its gain ratio down anyway (0.103 against 0.156 for `credit_history`).
3. **Thresholds vs intervals.** Iris, Trees tab. C4.5 needs 3 leaves; ID3 needs about a dozen because its intervals are fixed in advance. Change *ID3 intervals* from 4 to 2 or 8 and watch ID3's tree and accuracy change.
4. **Pruning.** Credit Risk, C4.5. Switch pruning off: the tree grows to about 140 nodes. Switch it back on and move CF from 0.50 down to 0.01; the *Pruned away* figure rises and the depth sweep flattens.
5. **Overfitting.** Credit Risk, Comparison tab, depth sweep. ID3 reaches 100 % training accuracy while its test accuracy peaks around depth 3; then set ID3's *Max depth* to 3 in the sidebar and compare again.
6. **Missing values.** Credit Risk, Missing values → *C4.5 native*. Leaf counts in the C4.5 tree become fractional. In Predict, set *credit_history* to *(unknown)*: ID3 takes the mode and follows one branch, while C4.5 follows every branch and blends the leaves.
7. **How much does the split matter?** Change the random seed a few times and watch the hold-out accuracy move; then compare with the cross-validation spread.
