# ============================================================================
# Paper:
# Author(s):
# Create Time: 01/07/2024
# ============================================================================

from pkg_manager import *
from para_manager import *

class BaseNN:
    def __init__(self):
        pass

    def fit(self, Predictors, Targets):
        self.nn_model_ = NearestNeighbors(n_neighbors=1, metric="euclidean")
        self.nn_model_.fit(Targets.cpu())
        self.Predictors = Predictors

        return self

    def transform(self, cfTarget):
        closestIdx = self.nn_model_.kneighbors(cfTarget, return_distance=False)
        cfPredictor = self.Predictors[closestIdx]

        return cfPredictor