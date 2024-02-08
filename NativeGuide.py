# ============================================================================
# Paper:
# Author(s):
# Create Time: 01/16/2023
# ============================================================================

from pkg_manager import *
from para_manager import *

class NativeGuide:
    def __init__(self):
        self.unlike_threshold = 0.3

    def fit(self, Predictors, Targets, thisTarget):
        self.Predictors = Predictors.squeeze()
        self.indices = torch.nonzero((Targets < thisTarget - self.unlike_threshold) | (Targets > thisTarget + self.unlike_threshold)).squeeze()[:,0]

        self.nn_model_ = KNeighborsTimeSeries(n_neighbors = 1, metric = "euclidean")
        selected_predictors = Predictors[self.indices]
        self.nn_model_.fit(selected_predictors.cpu())
        self.dist, self.index = self.nn_model_.kneighbors(Predictors.cpu(), return_distance=True)

        return self

    def transform(self, thisPredictor,  beta, i):
        insample_cf = self.Predictors[self.index[i]]
        generated_cf = dtw_barycenter_averaging([thisPredictor.squeeze().cpu(), insample_cf.squeeze().cpu()], weights=np.array([(1 - beta), beta])).squeeze() # return numpy

        return generated_cf
