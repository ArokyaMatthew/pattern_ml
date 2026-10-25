"""Regenerate the figures and the results table used in README.md.

Run from the repository root:

    python scripts/make_figures.py

Figures go to ``docs/images``; the results table is printed as Markdown.
Every number comes from the default workbench settings (30 % stratified
hold-out, seed 42), so the app shows the same values.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arbor import charts, metrics  # noqa: E402
from arbor.datasets import SAMPLES, load_sample  # noqa: E402
from arbor.experiment import C45, ID3, ExperimentConfig, cross_validate_models, depth_sweep, run_experiment  # noqa: E402
from arbor.preprocessing import identifier_columns, infer_kind, suggest_target  # noqa: E402

IMAGES = ROOT / "docs" / "images"


def default_config(frame, **overrides) -> ExperimentConfig:
    target = suggest_target(frame)
    features = tuple(c for c in frame.columns if c != target and c not in identifier_columns(frame))
    numeric = tuple(c for c in features if infer_kind(frame[c]) == "numeric")
    small = len(frame) < 30
    settings = {"test_size": 0.0 if small else 0.3, **overrides}
    return ExperimentConfig(target=target, features=features, numeric_features=numeric, **settings)


def save(figure, name: str) -> None:
    figure.savefig(IMAGES / name, dpi=200)
    print(f"wrote docs/images/{name}")


def figures() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    iris = load_sample("iris")
    experiment = run_experiment(iris, default_config(iris))
    for algorithm in (ID3, C45):
        run = experiment.runs[algorithm]
        cm = metrics.confusion(experiment.y_test, run.y_pred, experiment.labels)
        save(charts.confusion_matrix(cm, experiment.labels, title=f"{algorithm} on Iris"),
             f"confusion_{algorithm.lower().replace('.', '')}_iris.png")

    credit = load_sample("credit_risk")
    experiment = run_experiment(credit, default_config(credit))
    save(charts.metric_comparison(experiment.scores()), "comparison_credit_risk.png")
    save(charts.depth_curves(depth_sweep(experiment, 10)), "depth_sweep_credit_risk.png")


def results_table() -> None:
    print("\n| Dataset | Rows | Evaluation | ID3 accuracy | C4.5 accuracy | ID3 5-fold CV | C4.5 5-fold CV | ID3 leaves | C4.5 leaves |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for key, sample in SAMPLES.items():
        frame = load_sample(key)
        cfg = default_config(frame)
        experiment = run_experiment(frame, cfg)
        scores = experiment.scores()
        cv = cross_validate_models(frame, cfg, folds=5).groupby("algorithm")["accuracy"].agg(["mean", "std"])
        leaves = {name: run.model.n_leaves_ for name, run in experiment.runs.items()}
        evaluation = "30 % hold-out" if cfg.holdout else "training rows"
        print(
            f"| {sample.title} | {len(frame)} | {evaluation} "
            f"| {scores.loc[ID3, 'accuracy']:.3f} | {scores.loc[C45, 'accuracy']:.3f} "
            f"| {cv.loc[ID3, 'mean']:.3f} ± {cv.loc[ID3, 'std']:.3f} | {cv.loc[C45, 'mean']:.3f} ± {cv.loc[C45, 'std']:.3f} "
            f"| {leaves[ID3]} | {leaves[C45]} |"
        )


if __name__ == "__main__":
    figures()
    results_table()
