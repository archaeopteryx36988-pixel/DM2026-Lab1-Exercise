from typing import List, Union
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    """Factory function that constructs and returns tools for DTM processing."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
    ) -> dict:
        """Builds a document-term matrix (DTM) from text in session.dataframe using CountVectorizer.

        Constructs the feature matrix and vocabulary from the loaded DataFrame's
        'text' column, computes matrix sparsity stats, updates session state
        (session.feature_matrix, session.feature_names, session.artifacts["count_vectorizer"]),
        and sets per-document labels using session.dataframe['category_name'].

        Args:
            ngram_range: Lower and upper boundary of the range of n-values for
                different n-grams to be extracted. Formatted as List[int] [min_n, max_n]
                to ensure Gemini API JSON schema compatibility. Defaults to [1, 1].
            min_df: Minimum document frequency threshold (count or fraction). Defaults to 1.
            max_df: Maximum document frequency threshold (count or fraction). Defaults to 1.0.

        Returns:
            dict: Summary containing result_id, n_documents, n_vocabulary,
                non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            raise ValueError("No DataFrame loaded in session. Call load_dataset_tool first.")

        if "category_name" not in session.dataframe.columns:
            raise ValueError(
                "DataFrame missing 'category_name' column. Ensure dataset is loaded with category labels."
            )

        # Ensure text is treated as string, replacing NaN/empty with empty string
        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        # Convert list representation back to tuple for CountVectorizer
        ngram_tuple = (int(ngram_range[0]), int(ngram_range[1]))

        # Instantiate vectorizer without removing stop words
        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
            stop_words=None,
        )

        # Fit and transform texts into sparse matrix
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session attributes
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Set labels on session using category_name
        session.set_labels(session.dataframe["category_name"].values)

        # Calculate sparsity statistics
        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = (
            100.0 * (1.0 - (non_zero / total_elements)) if total_elements > 0 else 0.0
        )

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_vocabulary": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result(
            "build_dtm_tool",
            {"ngram_range": ngram_range, "min_df": min_df, "max_df": max_df},
            summary,
        )
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> dict:
        """Aggregates term frequencies across all documents in the document-term matrix.

        Computes column sums of session.feature_matrix, stores the complete term
        frequency table as a DataFrame in session.store_result's full_report, and
        returns a truncated summary suitable for LLM display.

        Args:
            top_n: Maximum number of top frequent terms to include in the returned
                summary dictionary (must be between 1 and 100). Defaults to 20.

        Returns:
            dict: Summary containing result_id, total_terms, total_frequency,
                and frequencies (dictionary mapping top terms to frequency counts).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "No DTM found in session. Call build_dtm_tool before running term_frequency_tool."
            )

        if not (1 <= top_n <= 100):
            raise ValueError("top_n must be an integer between 1 and 100.")

        # Compute column sums across all documents in feature_matrix
        counts = session.feature_matrix.sum(axis=0).A1.tolist()
        terms = list(session.feature_names)

        # Build complete DataFrame for full_report
        full_df = pd.DataFrame({"term": terms, "frequency": counts}).sort_values(
            by="frequency", ascending=False
        ).reset_index(drop=True)

        # Truncate summary frequencies strictly according to top_n
        top_df = full_df.head(top_n)
        summary_frequencies = dict(zip(top_df["term"], top_df["frequency"]))

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": len(terms),
            "total_frequency": int(sum(counts)),
            "frequencies": summary_frequencies,
        }

        # Store full report for visualize_result_tool or deep inspection
        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=full_df,
        )
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Renders a heatmap of a positional slice of the raw document-term matrix.

        Slices session.feature_matrix along documents and terms in original order,
        plots a seaborn heatmap assigned to session.pending_figure, and returns
        the sliced matrix and dimensions.

        Args:
            n_terms: Number of terms (columns) to include from positional start. Must be positive. Defaults to 20.
            n_documents: Number of documents (rows) to include from positional start. Must be positive. Defaults to 20.

        Returns:
            dict: Summary containing result_id, n_documents_shown, n_terms_shown,
                terms, and matrix_slice (nested list of counts).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "No DTM found in session. Call build_dtm_tool before running dtm_heatmap_tool."
            )

        if n_terms <= 0 or n_documents <= 0:
            raise ValueError("n_terms and n_documents must be positive integers.")

        total_rows, total_cols = session.feature_matrix.shape

        # Clamping display range to min(requested, actual, 50)
        actual_docs = min(n_documents, total_rows, 50)
        actual_terms = min(n_terms, total_cols, 50)

        # Slice sparse matrix first, then convert to dense array
        matrix_slice_sparse = session.feature_matrix[0:actual_docs, 0:actual_terms]
        matrix_slice_dense = matrix_slice_sparse.toarray()
        sliced_terms = session.feature_names[0:actual_terms]

        # Build figure and set on session.pending_figure (no plt.show() or plt.savefig())
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            matrix_slice_dense,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=[f"Doc {i}" for i in range(actual_docs)],
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap Slice ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        fig.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents_shown": actual_docs,
            "n_terms_shown": actual_terms,
            "terms": sliced_terms,
            "matrix_slice": matrix_slice_dense.tolist(),
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
