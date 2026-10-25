"""Arbor: ID3 and C4.5 decision tree classifiers, built from first principles.

The estimators follow scikit-learn's API::

    from arbor import C45Classifier

    model = C45Classifier(confidence=0.25).fit(X_train, y_train)
    model.predict(X_test)
    print(model.export_text())
"""

from arbor.c45 import C45Classifier
from arbor.id3 import ID3Classifier

__all__ = ["C45Classifier", "ID3Classifier"]
__version__ = "1.0.0"
