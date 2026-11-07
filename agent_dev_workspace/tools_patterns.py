import os
import tempfile
from typing import Literal
import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils.sparsefuncs import mean_variance_axis
from langchain_core.tools import tool

# PAMI imports
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.maximal import MaxFPGrowth
from PAMI.frequentPattern.topk import FAE


def make_tools(session):
    """Factory function that constructs and returns tools for frequent pattern mining."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "variance",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: int = 1,
        k: int = 10,
    ) -> dict:
        """Mines frequent patterns for a specific category using PAMI library algorithms.

        Constructs a fresh category-local DTM with English stop words removed,
        applies percentile feature filtering using sparse operations, converts surviving items
        into a PAMI-compatible transactional database, and extracts frequent itemsets.

        Args:
            category_name: Category label string to mine patterns for.
            filtering_method: Vocabulary filter approach ('variance', 'tfidf', or 'term_frequency'). Defaults to 'variance'.
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Minimum support threshold count for 'fpgrowth' and 'maxfpgrowth' (>= 1). Defaults to 1.
            k: Top-k pattern count for 'topk' algorithm (>= 1). Defaults to 10.

        Returns:
            dict: Summary containing result_id, category_name, filtering_method,
                algorithm, total_patterns, patterns (capped at top 50), and optional note.
        """
        if session.dataframe is None:
            raise ValueError("No DataFrame loaded in session. Call load_dataset_tool first.")

        if "category_name" not in session.dataframe.columns:
            raise ValueError("DataFrame missing 'category_name' column.")

        if min_sup < 1:
            raise ValueError("min_sup must be an integer >= 1.")

        if k < 1:
            raise ValueError("k must be an integer >= 1.")

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        # Helper to construct empty result summary and clear corresponding artifact
        def make_empty_result(note: str) -> dict:
            session.artifacts[artifact_key] = []
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "total_patterns": 0,
                "patterns": [],
                "note": note,
            }
            session.store_result(
                "mine_patterns_tool",
                {
                    "category_name": category_name,
                    "filtering_method": filtering_method,
                    "algorithm": algorithm,
                    "min_sup": min_sup,
                    "k": k,
                },
                summary,
            )
            return summary

        # Filter DataFrame for the requested category
        cat_df = session.dataframe[
            session.dataframe["category_name"] == category_name
        ]
        if cat_df.empty:
            raise ValueError(f"No documents found for category_name '{category_name}'.")

        texts = cat_df["text"].fillna("").astype(str).tolist()

        # Step 1: Build fresh, category-local DTM removing English stop words.
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm_sparse = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                return make_empty_result("no terms survived stopword removal")
            raise e

        n_docs, n_terms = dtm_sparse.shape
        if n_terms == 0:
            return make_empty_result("no terms survived stopword removal")

        # Step 2: Percentile filtering on sparse matrix
        if filtering_method == "variance":
            _, variances = mean_variance_axis(dtm_sparse, axis=0)
            p5 = scoreatpercentile(variances, 5)
            p95 = scoreatpercentile(variances, 95)
            mask = (variances > p5) & (variances < p95)
        elif filtering_method == "term_frequency":
            freqs = np.array(dtm_sparse.sum(axis=0)).ravel()
            p5 = scoreatpercentile(freqs, 5)
            p95 = scoreatpercentile(freqs, 95)
            mask = (freqs > p5) & (freqs < p95)
        elif filtering_method == "tfidf":
            tfidf_sparse = TfidfTransformer().fit_transform(dtm_sparse)
            tfidf_means = np.array(tfidf_sparse.mean(axis=0)).ravel()
            p20 = scoreatpercentile(tfidf_means, 20)
            mask = tfidf_means >= p20
        else:
            raise ValueError(f"Unsupported filtering_method '{filtering_method}'.")

        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            return make_empty_result("no terms survived filtering")

        kept_features = feature_names[kept_indices]
        filtered_dtm = dtm_sparse[:, kept_indices].tocsr()

        # Step 3: Build transaction file in PAMI tab-separated format efficiently from CSR sparse indices
        lines = []
        for i in range(n_docs):
            row_start = filtered_dtm.indptr[i]
            row_end = filtered_dtm.indptr[i + 1]
            if row_start < row_end:
                col_indices = filtered_dtm.indices[row_start:row_end]
                items = [kept_features[col_idx] for col_idx in col_indices]
                if items:
                    lines.append("\t".join(items))

        if not lines:
            return make_empty_result("transactional database is empty")

        # Write tab-delimited transaction lines to temporary file for PAMI miners
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".txt", encoding="utf-8"
        ) as tmp:
            tmp.write("\n".join(lines) + "\n")
            tmp_filename = tmp.name

        # Step 4: Run requested PAMI algorithm
        parsed_patterns = []
        try:
            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(iFile=tmp_filename, minSup=min_sup, sep="\t")
                miner.mine()
                raw_patterns = miner.getPatterns()
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(iFile=tmp_filename, minSup=min_sup, sep="\t")
                miner.mine()
                raw_patterns = miner.getPatterns()
            elif algorithm == "topk":
                miner = FAE.FAE(iFile=tmp_filename, k=k, sep="\t")
                miner.mine()
                raw_patterns = miner.getPatterns()
            else:
                raise ValueError(f"Unsupported algorithm '{algorithm}'.")

            # Helper to parse pattern keys into clean list of item strings
            for pattern, sup in raw_patterns.items():
                if isinstance(pattern, (list, tuple)):
                    items = [str(it).strip() for it in pattern if str(it).strip()]
                elif isinstance(pattern, str):
                    items = [
                        it.strip() for it in pattern.split("\t") if it.strip()
                    ]
                else:
                    items = [str(pattern).strip()]

                if items:
                    parsed_patterns.append(
                        {"pattern": items, "support": int(sup)}
                    )

        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)

        # Handle case where mining completes successfully but produces zero patterns
        if not parsed_patterns:
            return make_empty_result("no patterns found")

        # Sort patterns by support descending
        parsed_patterns.sort(key=lambda x: x["support"], reverse=True)

        total_patterns = len(parsed_patterns)
        summary_patterns = parsed_patterns[:50]

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "total_patterns": total_patterns,
            "patterns": summary_patterns,
        }

        # Store complete pattern list in session.artifacts
        session.artifacts[artifact_key] = parsed_patterns

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )
        return summary

    return [mine_patterns_tool]
