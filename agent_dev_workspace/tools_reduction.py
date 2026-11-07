import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, Optional, Dict, Any
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try importing UMAP if available
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def make_tools(session):
    """
    Factory function that creates tools for dimensionality reduction and label binarization.
    """

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Reduces feature_matrix dimensions using PCA, t-SNE, or UMAP, plots the 2D scatter plot
        colored by category, and stores the plot in session.pending_figure.

        Args:
            method: Reduction algorithm ("pca", "tsne", or "umap"). Default is "pca".
            n_components: Target dimensions (default 2, must be >= 2 for scatter plot).
            perplexity: Perplexity hyperparameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors for UMAP (default 15).
            random_state: Random seed for reproducibility.

        Returns:
            Dict containing result_id, method, coordinates summary, and status.
        """
        if n_components < 2:
            raise ValueError(f"n_components must be at least 2 for scatter plotting, got {n_components}.")

        if session.feature_matrix is None:
            return {"error": "No feature matrix found in session. Please run build_dtm_tool first."}

        # Convert sparse matrix to dense array if needed
        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()

        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)

        elif method == "umap":
            if not HAS_UMAP:
                return {"error": "umap-learn package is not installed."}

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(X)
        else:
            return {"error": f"Unsupported method: {method}"}

        # Store reduced coordinates in session artifacts
        session.artifacts[f"reduced_coords_{method}"] = coords.tolist()

        # Build scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Retrieve categories or labels for coloring
        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values

        if labels is not None:
            sns.scatterplot(
                x=coords[:, 0],
                y=coords[:, 1],
                hue=labels,
                palette="tab10",
                ax=ax,
                s=70
            )
            ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], c="blue", alpha=0.7)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        # Set figure on session.pending_figure per spec
        session.pending_figure = fig

        # Store result summary
        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_samples": coords.shape[0],
            "reduced_coordinates": coords.tolist(),
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary=summary,
            full_report={"coordinates": coords.tolist()}
        )

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """
        One-hot encodes document category labels using session.labels or session.dataframe['category_name'].
        Generates one column per unique category (e.g. 2 categories produce 2 columns).

        Returns:
            Dict containing result_id, categories, n_samples, and one_hot_labels matrix.
        """
        labels = None
        if session.labels is not None and len(session.labels) > 0:
            labels = list(session.labels)
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].tolist()

        if not labels:
            return {"error": "No category labels found in session. Call load_dataset_tool or build_dtm_tool first."}

        # Get sorted unique category names
        unique_categories = sorted(list(set(labels)))
        if not unique_categories:
            return {"error": "Category labels are empty."}

        # Perform one-hot encoding (always 1 column per unique category)
        one_hot_matrix = []
        for label in labels:
            row = [1 if label == cat else 0 for cat in unique_categories]
            one_hot_matrix.append(row)

        # Store in session artifacts
        session.artifacts["binarized_labels"] = {
            "categories": unique_categories,
            "one_hot_matrix": one_hot_matrix,
        }

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "categories": unique_categories,
            "n_samples": len(labels),
            "one_hot_labels": one_hot_matrix,
        }

        # Build full report DataFrame
        full_df = pd.DataFrame(one_hot_matrix, columns=unique_categories)

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
            full_report=full_df,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
