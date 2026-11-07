import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index1: int, doc_index2: int) -> dict:
        """Computes the cosine similarity between two documents in session.dataframe.

        Args:
            doc_index1: 0-based integer index of the first document.
            doc_index2: 0-based integer index of the second document.

        Returns:
            A summary dictionary containing result_id, doc_index1, doc_index2,
            and the computed cosine_similarity, or an error description.
        """
        args = {"doc_index1": doc_index1, "doc_index2": doc_index2}

        if session.dataframe is None:
            return {"error": "session.dataframe is None. Please load a dataset first."}

        if "text" not in session.dataframe.columns:
            return {"error": "session.dataframe missing required 'text' column."}

        vectorizer = session.artifacts.get("count_vectorizer") if session.artifacts else None
        if vectorizer is None:
            return {"error": "Missing fitted CountVectorizer in session.artifacts['count_vectorizer']. Please run build_dtm_tool first."}

        n_docs = len(session.dataframe)
        if doc_index1 < 0 or doc_index1 >= n_docs:
            return {"error": f"doc_index1 out of bounds: {doc_index1} (valid range: 0 to {n_docs - 1})."}
        if doc_index2 < 0 or doc_index2 >= n_docs:
            return {"error": f"doc_index2 out of bounds: {doc_index2} (valid range: 0 to {n_docs - 1})."}

        text1 = str(session.dataframe.iloc[doc_index1]["text"]) if session.dataframe.iloc[doc_index1]["text"] is not None else ""
        text2 = str(session.dataframe.iloc[doc_index2]["text"]) if session.dataframe.iloc[doc_index2]["text"] is not None else ""

        vecs = vectorizer.transform([text1, text2])

        norm1 = np.linalg.norm(vecs[0].toarray())
        norm2 = np.linalg.norm(vecs[1].toarray())

        if norm1 == 0 or norm2 == 0:
            sim_val = 0.0
        else:
            sim_val = float(cosine_similarity(vecs[0], vecs[1])[0][0])

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "cosine_similarity": sim_val,
        }

        session.store_result("cosine_similarity_tool", args, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Computes the Pearson correlation matrix for the top 20 terms by variance in the global DTM.

        Returns:
            A summary dictionary containing result_id, feature_names, and correlation_matrix,
            or an error description. Also sets session.pending_figure to the correlation heatmap.
        """
        args = {}

        if session.feature_matrix is None:
            return {"error": "session.feature_matrix is None. Please build the DTM first."}

        if session.feature_names is None or len(session.feature_names) == 0:
            return {"error": "session.feature_names is missing or empty."}

        # Convert to dense array if sparse
        dense_fm = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        if dense_fm.shape[0] < 2:
            return {"error": f"Requires at least 2 documents to compute correlation, got {dense_fm.shape[0]}."}

        if dense_fm.shape[1] == 0:
            return {"error": "session.feature_matrix has 0 columns."}

        # Calculate variance per feature column
        variances = np.var(dense_fm, axis=0)

        # Select top min(20, n_features) by variance descending
        n_features = dense_fm.shape[1]
        k = min(20, n_features)

        # Get indices sorted by variance descending
        top_indices = np.argsort(variances)[::-1][:k]

        selected_names = [session.feature_names[i] for i in top_indices]
        selected_matrix = dense_fm[:, top_indices]

        # Handle k=1 specially
        if k == 1:
            var_val = variances[top_indices[0]]
            if var_val > 0:
                corr_matrix_np = np.array([[1.0]])
            else:
                corr_matrix_np = np.array([[np.nan]])
        else:
            # Compute Pearson correlation matrix across columns (rowvar=False)
            corr_matrix_np = np.corrcoef(selected_matrix, rowvar=False)

        # Plot correlation heatmap with mask for NaN
        fig, ax = plt.subplots(figsize=(max(6, k * 0.4), max(5, k * 0.3)))
        mask = np.isnan(corr_matrix_np)
        sns.heatmap(
            corr_matrix_np,
            xticklabels=selected_names,
            yticklabels=selected_names,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            cbar=True,
            mask=mask
        )
        ax.set_title("Top Feature Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig

        # Convert NaN to None for JSON serializable summary without fabricating zeros
        corr_matrix_list = [
            [None if np.isnan(val) else float(val) for val in row]
            for row in corr_matrix_np
        ]

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "feature_names": selected_names,
            "correlation_matrix": corr_matrix_list,
        }

        session.store_result("feature_correlation_matrix_tool", args, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
