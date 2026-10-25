"""Method tab: the theory behind the two algorithms, as implemented here."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components import section

COMPARISON = pd.DataFrame(
    {
        "": [
            "Split criterion",
            "Attribute types",
            "Split shape",
            "Re-using an attribute",
            "Missing values",
            "Stopping",
            "Pruning",
        ],
        "ID3 (Quinlan, 1986)": [
            "Information gain",
            "Categorical only; numbers are discretised first",
            "One branch per value",
            "Never on the same path",
            "Imputed before training",
            "Pure node, no attributes left, or a pre-pruning limit",
            "None",
        ],
        "C4.5 (Quinlan, 1993)": [
            "Gain ratio, among attributes with at least average gain",
            "Categorical and continuous",
            "One branch per value, or binary  A ≤ t",
            "Continuous attributes can be split again",
            "Fractional cases, in training and prediction",
            "Pure node, or fewer than 2 × m cases, or no admissible split",
            "Error-based subtree replacement (CF)",
        ],
    }
)


def render() -> None:
    section(
        "ID3 and C4.5 at a glance",
        eyebrow="Method",
        description="Both algorithms grow a tree top-down and greedily: at each node they pick the attribute that "
        "best separates the classes, split the cases, and repeat on every branch. They differ in how they measure "
        "'best' and in what they can handle.",
        first=True,
    )
    st.dataframe(COMPARISON, hide_index=True, width="stretch")

    left, right = st.columns(2, gap="large")
    with left:
        section("Entropy and information gain", eyebrow="ID3")
        st.markdown("Entropy measures how mixed the classes are in a set of cases $S$, in bits:")
        st.latex(r"H(S) = -\sum_{i=1}^{k} p_i \log_2 p_i")
        st.markdown("Information gain is the drop in entropy after splitting $S$ on attribute $A$:")
        st.latex(r"\operatorname{Gain}(S, A) = H(S) - \sum_{v \in \operatorname{values}(A)} \frac{|S_v|}{|S|}\, H(S_v)")
        st.markdown(
            "ID3 splits on the attribute with the highest gain. Gain is biased toward attributes with many "
            "values: an ID column would split every case into its own pure leaf and score the maximum."
        )
        section("Discretisation for ID3")
        st.markdown(
            "ID3 cannot test `x ≤ t`, so each numeric attribute becomes a handful of intervals before training. "
            "**Equal frequency** puts roughly the same number of training cases in each interval, "
            "**equal width** cuts the range evenly, **k-means** places cuts between 1-D clusters. "
            "Cut points are learned on the training split only."
        )

    with right:
        section("Split information and gain ratio", eyebrow="C4.5")
        st.markdown("C4.5 divides gain by the entropy of the split itself, which grows with the number of branches:")
        st.latex(r"\operatorname{SplitInfo}(S, A) = -\sum_{v} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}")
        st.latex(r"\operatorname{GainRatio}(S, A) = \frac{\operatorname{Gain}(S, A)}{\operatorname{SplitInfo}(S, A)}")
        st.markdown(
            "Because a tiny split information can inflate the ratio, only attributes whose gain is at least the "
            "average gain of all candidates are admitted."
        )
        section("Continuous attributes")
        st.markdown(
            "Cases are sorted on the attribute; every boundary between two distinct values is a candidate "
            "threshold $t$ (the midpoint is used). The cut with the highest gain wins, and, as in C4.5 Release 8, "
            r"$\log_2(\text{cuts}) / |S|$ bits are subtracted for having searched that many cuts."
        )

    left, right = st.columns(2, gap="large")
    with left:
        section("Missing values", eyebrow="C4.5")
        st.markdown(
            "With *C4.5 native* selected, gain is computed on cases whose value is known and multiplied by the "
            "known fraction $F$; split information counts *unknown* as one more outcome:"
        )
        st.latex(r"\operatorname{Gain}(S, A) = F \cdot \big(H(S_{\text{known}}) - \textstyle\sum_v \frac{|S_v|}{|S_{\text{known}}|} H(S_v)\big)")
        st.markdown(
            "A case with an unknown value goes down **every** branch with weight proportional to the branch's "
            "share of known cases. At prediction time it does the same and the leaf class distributions are blended."
        )
    with right:
        section("Error-based pruning", eyebrow="C4.5")
        st.markdown(
            "A leaf with $N$ training cases and $E$ errors is assumed to err, on unseen data, at the upper limit "
            "$U_{CF}(E, N)$ of a binomial confidence interval. Using the normal approximation with $z = \\Phi^{-1}(1-CF)$:"
        )
        st.latex(
            r"U_{CF}(E,N) = \frac{E + \tfrac12 + \tfrac{z^2}{2} + z\sqrt{(E+\tfrac12)\big(1-\tfrac{E+1/2}{N}\big) + \tfrac{z^2}{4}}}{N + z^2}"
        )
        st.markdown(
            "Working bottom-up, a subtree is replaced by a leaf when the leaf's estimated errors $N \\cdot U_{CF}$ "
            "are no more than the sum over the subtree's leaves (plus 0.1). Lower CF means larger estimates and "
            "harder pruning. Subtree raising is not implemented."
        )

    section("Evaluation", eyebrow="Metrics")
    st.markdown(
        "With $TP_c$, $FP_c$, $FN_c$ counted per class $c$ from the confusion matrix:"
    )
    st.latex(
        r"\text{Precision}_c = \frac{TP_c}{TP_c + FP_c} \qquad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}"
        r"\qquad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}"
    )
    st.markdown(
        "**Macro** averages treat every class equally; **weighted** averages scale each class by its support. "
        "Accuracy is the share of all cases classified correctly."
    )

    section("References", eyebrow="Further reading")
    st.markdown(
        "- J. R. Quinlan. *Induction of decision trees.* Machine Learning 1(1), 81–106, 1986.\n"
        "- J. R. Quinlan. *C4.5: Programs for Machine Learning.* Morgan Kaufmann, 1993.\n"
        "- J. R. Quinlan. *Improved use of continuous attributes in C4.5.* JAIR 4, 77–90, 1996.\n"
        "- T. M. Mitchell. *Machine Learning*, chapter 3. McGraw-Hill, 1997."
    )
