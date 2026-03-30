"""
Handcrafted radiomics + survival analysis models.

Classical baselines for survival prediction:
- Cox Proportional Hazards (lifelines)
- Random Survival Forest (scikit-survival)
- Feature selection: LASSO, mutual information, variance threshold
- C-index evaluation and Kaplan-Meier curves
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
except ImportError:
    raise ImportError("Please install lifelines: pip install lifelines")

try:
    from sksurv.ensemble import RandomSurvivalForest as SKSurvRandomSurvivalForest
    from sksurv.util import Surv
except ImportError:
    raise ImportError("Please install scikit-survival: pip install scikit-survival")

from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class FeatureSelector:
    """Feature selection for radiomics features."""

    def __init__(
        self,
        method: str = "variance",
        n_features: Optional[int] = None,
        threshold: float = 0.01,
    ):
        """Initialize feature selector.

        Args:
            method: 'variance', 'lasso', or 'mutual_info'
            n_features: Number of features to keep
            threshold: Variance threshold for variance-based selection
        """
        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.selected_features = None
        self.selector = None

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> List[str]:
        """Fit feature selector.

        Args:
            X: Feature matrix (num_samples, num_features)
            y: Target variable (for LASSO / mutual info)

        Returns:
            List of selected feature names
        """
        if self.method == "variance":
            self.selector = VarianceThreshold(threshold=self.threshold)
            self.selector.fit(X)
            mask = self.selector.get_support()
            self.selected_features = np.where(mask)[0]

        elif self.method == "lasso":
            if y is None:
                raise ValueError("LASSO requires target variable y")
            # Store scaler to apply same transform to test data (no leakage)
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            lasso = Lasso(alpha=0.01, max_iter=10000)
            lasso.fit(X_scaled, y)

            mask = lasso.coef_ != 0
            self.selected_features = np.where(mask)[0]

        elif self.method == "mutual_info":
            if y is None:
                raise ValueError("Mutual information requires target variable y")
            selector = SelectKBest(
                mutual_info_classif,
                k=self.n_features or min(20, X.shape[1]),
            )
            selector.fit(X, y)
            self.selected_features = selector.get_support(indices=True)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self.selected_features.tolist()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using selected features.

        Args:
            X: Feature matrix

        Returns:
            Transformed matrix with selected features only
        """
        if self.selected_features is None:
            raise RuntimeError("Selector not fitted yet")
        return X[:, self.selected_features]


class CoxProportionalHazards:
    """Cox Proportional Hazards model for survival analysis.

    Uses lifelines library.
    """

    def __init__(
        self,
        penalizer: float = 0.0,
    ):
        """Initialize Cox model.

        Args:
            penalizer: L2 penalty strength
        """
        self.penalizer = penalizer
        self.model = CoxPHFitter(penalizer=penalizer)
        self.feature_names = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """Fit Cox model.

        Args:
            X: Feature matrix (num_samples, num_features)
            T: Time to event (num_samples,)
            E: Event indicator (num_samples,) - 1 for event, 0 for censored
            feature_names: Optional feature names
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Create DataFrame for lifelines
        df = pd.DataFrame(X, columns=self.feature_names)
        df['T'] = T
        df['E'] = E

        # Fit model
        self.model.fit(df, duration_col='T', event_col='E')
        self.is_fitted = True

    def predict(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict survival probability or risk score.

        Args:
            X: Feature matrix
            times: Optional times to evaluate survival probability

        Returns:
            Risk scores (baseline) or survival probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        df = pd.DataFrame(X, columns=self.feature_names)

        if times is not None:
            # Survival probability at times
            return self.model.predict_survival_function(df, times=times).values.T
        else:
            # Partial hazard / risk score
            return self.model.predict_partial_hazard(df).values

    def concordance_index(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> float:
        """Compute C-index on test data.

        Args:
            X: Feature matrix
            T: Time to event
            E: Event indicator

        Returns:
            C-index (0.5 = random, 1.0 = perfect)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        risk_scores = self.predict(X)
        c_index = concordance_index(T, -risk_scores, E)

        return c_index

    def get_feature_importance(self, top_k: int = 10) -> pd.DataFrame:
        """Get feature importance (hazard ratios).

        Args:
            top_k: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        summary = self.model.summary
        summary_sorted = summary.sort_values('exp(coef)', ascending=False)

        return summary_sorted.head(top_k)


class RandomSurvivalForest:
    """Random Survival Forest for survival analysis.

    Uses scikit-survival library.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
    ):
        """Initialize RSF.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            random_state: Random seed
        """
        self.model = SKSurvRandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        self.feature_names = None
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """Fit RSF model.

        Args:
            X: Feature matrix
            T: Time to event
            E: Event indicator (1 for event, 0 for censored)
            feature_names: Optional feature names
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Create structured array for scikit-survival
        y = np.array([(bool(e), t) for e, t in zip(E, T)],
                     dtype=[('event', bool), ('time', float)])

        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores.

        Args:
            X: Feature matrix

        Returns:
            Risk scores (higher = higher risk)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        return self.model.predict(X)

    def concordance_index(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
    ) -> float:
        """Compute C-index on test data.

        Args:
            X: Feature matrix
            T: Time to event
            E: Event indicator

        Returns:
            C-index
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        # Create structured array
        y = np.array([(bool(e), t) for e, t in zip(E, T)],
                     dtype=[('event', bool), ('time', float)])

        # Compute C-index from scikit-survival
        from sksurv.metrics import concordance_index_censored

        risk_scores = self.predict(X)
        c_index = concordance_index_censored(E, T, risk_scores)[0]

        return c_index

    def get_feature_importance(self, top_k: int = 10) -> Dict[str, float]:
        """Get feature importance.

        Args:
            top_k: Number of top features

        Returns:
            Dictionary of feature importances
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        importances = self.model.feature_importances_
        feature_importance_dict = {
            name: imp for name, imp in zip(self.feature_names, importances)
        }

        # Sort by importance
        sorted_dict = dict(sorted(
            feature_importance_dict.items(),
            key=lambda x: x[1],
            reverse=True,
        ))

        return {k: v for k, v in list(sorted_dict.items())[:top_k]}


class KaplanMeierAnalysis:
    """Kaplan-Meier survival curve analysis."""

    def __init__(self):
        """Initialize KM fitter."""
        self.kmf = KaplanMeierFitter()

    def fit_and_plot(
        self,
        T: np.ndarray,
        E: np.ndarray,
        groups: Optional[np.ndarray] = None,
        label: str = "survival",
        ax=None,
    ):
        """Fit and plot KM curves.

        Args:
            T: Time to event
            E: Event indicator
            groups: Optional group labels for stratified analysis
            label: Label for curve
            ax: Optional matplotlib axis

        Returns:
            KM fitter object or matplotlib axis
        """
        self.kmf.fit(T, E, label=label)

        if ax is None:
            self.kmf.plot_survival_function()
        else:
            self.kmf.plot_survival_function(ax=ax)

        if groups is not None:
            # Stratified analysis
            unique_groups = np.unique(groups)
            kmfs = {}

            for group in unique_groups:
                mask = groups == group
                kmf = KaplanMeierFitter()
                kmf.fit(T[mask], E[mask], label=f"{label}_{group}")
                kmfs[group] = kmf

            return kmfs

        return self.kmf

    def get_median_survival(self) -> float:
        """Get median survival time.

        Returns:
            Median survival time
        """
        return self.kmf.median_survival_time_

    def survival_at_time(self, t: float) -> float:
        """Get survival probability at time t.

        Args:
            t: Time point

        Returns:
            Survival probability
        """
        return self.kmf.survival_function_at_times(t).values[0]
