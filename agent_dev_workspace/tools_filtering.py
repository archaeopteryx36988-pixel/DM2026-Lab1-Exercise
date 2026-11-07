import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.utils.sparsefuncs import mean_variance_axis
from langchain_core.tools import tool


def make_tools(session):
    """Factory function that constructs and returns tools for feature filtering."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filters DTM features by population variance threshold.

        Computes numerically stable population variance (ddof=0) for each feature
        in session.feature_matrix using sklearn.utils.sparsefuncs.mean_variance_axis
        without converting the sparse matrix to dense. Stores a sorted full_report
        DataFrame (first column: 'feature', second column: 'variance') in session.store_result,
        and saves kept features in session.artifacts['variance_filter'].

        Args:
            threshold: Minimum population variance threshold. Features with variance
                strictly greater than this value are retained. Defaults to 0.0.

        Returns:
            dict: Summary containing result_id, total_features, kept_features_count,
                removed_features_count, threshold, and top_variances.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "No DTM found in session. Call build_dtm_tool before running variance_filter_tool."
            )

        if threshold < 0.0:
            raise ValueError("threshold must be a non-negative float.")

        X = session.feature_matrix
        feature_names = session.feature_names
        n_docs = X.shape[0]

        if n_docs == 0:
            raise ValueError("Cannot compute variance on an empty feature matrix.")

        # Compute numerically stable population variance (ddof=0) for sparse matrix
        _, variances = mean_variance_axis(X, axis=0)

        # Build full_report DataFrame sorted descending by variance
        full_df = pd.DataFrame(
            {"feature": feature_names, "variance": variances}
        ).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Filter features with variance strictly greater than threshold
        kept_df = full_df[full_df["variance"] > threshold]
        kept_features = kept_df["feature"].tolist()

        # Update session.artifacts without mutating original feature_matrix or feature_names
        session.artifacts["variance_filter"] = kept_features

        # Limit top_variances summary payload size to top 20
        top_df = full_df.head(20)
        top_variances = dict(zip(top_df["feature"], top_df["variance"].round(6)))

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "total_features": len(feature_names),
            "kept_features_count": len(kept_features),
            "removed_features_count": len(feature_names) - len(kept_features),
            "threshold": threshold,
            "top_variances": top_variances,
        }

        # Store full report for visualize_result_tool (first col: feature, second col: variance)
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=full_df,
        )
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filters features by Pearson correlation r > threshold against a target class label.

        Computes Pearson correlation coefficient between each feature's count and a
        binary indicator vector for target_class. Handles zero-variance features safely
        without producing NaN, and clamps values to [-1.0, 1.0]. Stores full_report DataFrame
        (first column: 'feature', second column: 'pearson_r') sorted descending by pearson_r,
        and updates session.artifacts['pearson_filter'].

        Args:
            target_class: Target category label string to correlate against.
            threshold: Minimum Pearson correlation threshold. Features with pearson_r
                strictly greater than this value are retained. Defaults to 0.0.

        Returns:
            dict: Summary containing result_id, target_class, total_features,
                kept_features_count, removed_features_count, threshold, and top_correlations.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "No DTM found in session. Call build_dtm_tool before running pearson_filter_tool."
            )

        if session.labels is None or len(session.labels) == 0:
            raise ValueError(
                "No category labels found in session. Ensure dataset is loaded with labels."
            )

        if target_class not in session.labels:
            raise ValueError(
                f"target_class '{target_class}' not found in session.labels."
            )

        X = session.feature_matrix
        feature_names = session.feature_names
        n_docs = X.shape[0]

        if n_docs == 0:
            raise ValueError("Cannot compute correlation on an empty feature matrix.")

        # Create binary target indicator vector y (1 if label == target_class, else 0)
        y = (session.labels == target_class).astype(float)
        y_mean = np.mean(y)
        y_diff = y - y_mean
        y_std = np.sqrt(np.sum(y_diff**2))

        if y_std == 0:
            # If target vector has zero variance, correlation with all features is 0
            correlations = np.zeros(len(feature_names))
        else:
            # Compute feature means and standard deviations in a sparse manner
            means, variances = mean_variance_axis(X, axis=0)
            stds = np.sqrt(variances * n_docs)  # sqrt(sum((X - mean)^2))

            # Compute covariance: sum((X_ij - mean_j) * (y_i - mean_y)) = sum(X_ij * y_diff_i)
            # Matrix multiplication X.T @ y_diff efficiently calculates column sums weighted by y_diff
            covariances = np.array(X.T.dot(y_diff)).ravel()

            # Safely compute Pearson r, replacing zero-std divisions with 0.0
            correlations = np.zeros(len(feature_names))
            valid_mask = stds > 0
            correlations[valid_mask] = covariances[valid_mask] / (
                stds[valid_mask] * y_std
            )

        # Clip values to exact [-1.0, 1.0] interval to prevent floating point inaccuracies from exceeding theoretical bounds
        correlations = np.clip(correlations, -1.0, 1.0)

        # Build full_report DataFrame sorted descending by pearson_r
        full_df = pd.DataFrame(
            {"feature": feature_names, "pearson_r": correlations}
        ).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        # Filter features with pearson_r strictly greater than threshold (r > threshold)
        kept_df = full_df[full_df["pearson_r"] > threshold]
        kept_features = kept_df["feature"].tolist()

        # Update session.artifacts without mutating original feature_matrix, feature_names, or labels
        session.artifacts["pearson_filter"] = kept_features

        # Limit top_correlations summary payload size to top 20
        top_df = full_df.head(20)
        top_correlations = dict(zip(top_df["feature"], top_df["pearson_r"].round(6)))

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_features": len(feature_names),
            "kept_features_count": len(kept_features),
            "removed_features_count": len(feature_names) - len(kept_features),
            "threshold": threshold,
            "top_correlations": top_correlations,
        }

        # Store full report for visualize_result_tool (first col: feature, second col: pearson_r)
        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=full_df,
        )
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filters features by Spearman rank correlation r > threshold against a target class label.

        Computes Spearman rank correlation coefficient between each feature's count rank
        and a binary indicator rank vector for target_class. Handles zero-variance features
        and ties safely using scipy.stats.rankdata(method='average'). Stores full_report DataFrame
        (first column: 'feature', second column: 'spearman_r') sorted descending by spearman_r,
        and updates session.artifacts['spearman_filter'].

        Args:
            target_class: Target category label string to correlate against.
            threshold: Minimum Spearman correlation threshold. Features with spearman_r
                strictly greater than this value are retained. Defaults to 0.0.

        Returns:
            dict: Summary containing result_id, target_class, total_features,
                kept_features_count, removed_features_count, threshold, and top_correlations.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "No DTM found in session. Call build_dtm_tool before running spearman_filter_tool."
            )

        if session.labels is None or len(session.labels) == 0:
            raise ValueError(
                "No category labels found in session. Ensure dataset is loaded with labels."
            )

        if target_class not in session.labels:
            raise ValueError(
                f"target_class '{target_class}' not found in session.labels."
            )

        X = session.feature_matrix
        feature_names = session.feature_names
        n_docs, n_terms = X.shape

        if n_docs == 0:
            raise ValueError("Cannot compute correlation on an empty feature matrix.")

        # Create binary target indicator vector y (1 if label == target_class, else 0)
        y = (session.labels == target_class).astype(float)

        # Convert target vector y to ranks (handling ties with average method)
        y_ranks = rankdata(y, method="average")
        y_ranks_mean = np.mean(y_ranks)
        y_ranks_diff = y_ranks - y_ranks_mean
        y_ranks_std = np.sqrt(np.sum(y_ranks_diff**2))

        if y_ranks_std == 0:
            # If target ranks vector has zero variance, correlation with all features is 0
            correlations = np.zeros(n_terms)
        else:
            # Efficiently compute ranks for sparse DTM column by column
            # (only converting individual 1D column vectors to dense for rankdata)
            correlations = np.zeros(n_terms)
            for j in range(n_terms):
                col_data = X[:, j].toarray().ravel()
                col_ranks = rankdata(col_data, method="average")
                col_ranks_diff = col_ranks - np.mean(col_ranks)
                col_ranks_std = np.sqrt(np.sum(col_ranks_diff**2))

                if col_ranks_std > 0:
                    r = np.sum(col_ranks_diff * y_ranks_diff) / (
                        col_ranks_std * y_ranks_std
                    )
                    correlations[j] = r
                else:
                    correlations[j] = 0.0

        # Clip values to exact [-1.0, 1.0] interval to prevent floating point inaccuracies from exceeding theoretical bounds
        correlations = np.clip(correlations, -1.0, 1.0)

        # Build full_report DataFrame sorted descending by spearman_r
        full_df = pd.DataFrame(
            {"feature": feature_names, "spearman_r": correlations}
        ).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        # Filter features with spearman_r strictly greater than threshold (r > threshold)
        kept_df = full_df[full_df["spearman_r"] > threshold]
        kept_features = kept_df["feature"].tolist()

        # Update session.artifacts without mutating original feature_matrix, feature_names, or labels
        session.artifacts["spearman_filter"] = kept_features

        # Limit top_correlations summary payload size to top 20
        top_df = full_df.head(20)
        top_correlations = dict(zip(top_df["feature"], top_df["spearman_r"].round(6)))

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_features": len(feature_names),
            "kept_features_count": len(kept_features),
            "removed_features_count": len(feature_names) - len(kept_features),
            "threshold": threshold,
            "top_correlations": top_correlations,
        }

        # Store full report for visualize_result_tool (first col: feature, second col: spearman_r)
        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=full_df,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
