import numpy as np
import pickle
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)

class RegimeHMM:
    def __init__(self, n_states: int = 3, covariance_type: str = "diag"):
        """
        Hidden Markov Model to detect stochastic market regimes.
        State 0: Volatile
        State 1: Trending
        State 2: Mean-Reverting
        (Note: the mapping of physical states to names requires post-hoc analysis based on emission means)
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type=covariance_type, 
            n_iter=100, 
            random_state=42
        )
        self.is_fitted = False
        
        # Human readable labels mapping (determined after fitting)
        self.state_labels = {
            0: "STATE_0",
            1: "STATE_1",
            2: "STATE_2"
        }

    def fit(self, features: np.ndarray):
        """
        Args:
            features: [n_samples, n_features] array. 
                      Features typically include realized volatility and autocorrelation.
        """
        logger.info(f"Fitting RegimeHMM with {self.n_states} states on {len(features)} samples.")
        self.model.fit(features)
        self.is_fitted = True
        
        # Sort states by volatility (assuming feature index 0 is realized volatility)
        # to assign consistent labels
        means = self.model.means_[:, 0]
        sorted_indices = np.argsort(means)
        
        # 0: lowest vol (Mean-Reverting/Quiet)
        # 1: medium vol (Trending)
        # 2: highest vol (Volatile)
        self.state_labels[sorted_indices[0]] = "MEAN_REV"
        self.state_labels[sorted_indices[1]] = "TRENDING"
        self.state_labels[sorted_indices[2]] = "VOLATILE"
        
        logger.info(f"HMM fitted. State Mapping: {self.state_labels}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts the hidden state sequence.
        """
        if not self.is_fitted:
            raise ValueError("RegimeHMM must be fitted before calling predict().")
        return self.model.predict(features)
        
    def predict_regime_labels(self, features: np.ndarray) -> list[str]:
        states = self.predict(features)
        return [self.state_labels[s] for s in states]

    def save(self, filepath: str):
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        with open(filepath, 'wb') as f:
            pickle.dump({
                "model": self.model,
                "state_labels": self.state_labels,
                "n_states": self.n_states
            }, f)
        logger.info(f"RegimeHMM saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "RegimeHMM":
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        instance = cls(n_states=data["n_states"])
        instance.model = data["model"]
        instance.state_labels = data["state_labels"]
        instance.is_fitted = True
        return instance
