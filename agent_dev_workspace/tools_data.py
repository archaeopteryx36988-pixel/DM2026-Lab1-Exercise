import os
import json
import math
from datetime import datetime, date
from pathlib import Path
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import StrictInt
from langchain_core.tools import tool

def _sanitize_for_json(obj):
    """Recursively converts pandas/numpy/datetime types into JSON-serializable native Python objects.
    Converts NaN, None, pd.NA, NaT -> None; inf/-inf -> "Infinity"/"-Infinity"; datetimes -> ISO strings.
    """
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)

    if isinstance(obj, (int, np.integer)):
        return int(obj)

    if isinstance(obj, (float, np.floating)):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return None
        return float(obj)

    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())

    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]

    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}

    if obj is None or pd.isna(obj):
        return None

    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()

    return str(obj)


def make_tools(session):
    @tool
    def tokenize_tool() -> dict:
        """Tokenizes each document's text in session.dataframe into unigrams using NLTK word_tokenize.

        Reads session.dataframe['text'], converts text to lowercase, applies nltk.word_tokenize,
        stores token lists in session.dataframe['unigrams'], and returns summary stats.

        Returns:
            dict: Summary containing result_id, n_documents, total_tokens, and token_preview,
                or an error description.
        """
        args = {}

        if session.dataframe is None:
            return {"error": "session.dataframe is None. Please load a dataset first."}

        if "text" not in session.dataframe.columns:
            return {"error": "session.dataframe missing required 'text' column."}

        unigrams_list = []
        total_tokens = 0

        for val in session.dataframe["text"]:
            if pd.isna(val) or str(val).strip() == "":
                tokens = []
            else:
                try:
                    text_str = str(val).lower()
                    tokens = nltk.word_tokenize(text_str)
                except LookupError:
                    return {
                        "error": "NLTK tokenizer resource not found. "
                                 "Please ensure required NLTK data (e.g. 'punkt') is installed."
                    }

            unigrams_list.append(tokens)
            total_tokens += len(tokens)

        # Update dataframe and artifacts only after all tokenization completes successfully
        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        result_id = session.next_result_id("tokenize")
        
        # Token preview for the first 3 documents
        preview = unigrams_list[:3]

        summary = {
            "result_id": result_id,
            "n_documents": len(unigrams_list),
            "total_tokens": total_tokens,
            "token_preview": preview,
        }

        session.store_result("tokenize_tool", args, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """Lists files and directories in a specific subdirectory of the project repository.

        Args:
            subdirectory: Subdirectory path relative to the repo root. Defaults to "" (repo root).

        Returns:
            dict: Summary containing result_id, subdirectory, directories, and files lists,
                or an error description.
        """
        args = {"subdirectory": subdirectory}

        # Locate Repo root by searching upwards for pyproject.toml
        current = Path.cwd().resolve()
        repo_root = None
        for p in [current] + list(current.parents):
            if (p / "pyproject.toml").exists():
                repo_root = p
                break

        if repo_root is None:
            return {"error": "Could not locate repository root (pyproject.toml not found in current path or parents)."}

        if not subdirectory or subdirectory.strip() == ".":
            target_path = repo_root
        else:
            target_path = (repo_root / subdirectory).resolve()

        # Prevent directory traversal outside repo root
        try:
            target_path.relative_to(repo_root)
        except ValueError:
            return {"error": f"Access denied: Path '{subdirectory}' resolves outside repository root."}

        if not target_path.exists():
            return {"error": f"Path '{subdirectory}' does not exist."}

        if not target_path.is_dir():
            return {"error": f"Path '{subdirectory}' is not a directory."}

        directories = []
        files = []

        try:
            for entry in os.scandir(target_path):
                if entry.is_dir():
                    directories.append(entry.name)
                elif entry.is_file():
                    files.append(entry.name)
        except PermissionError:
            return {"error": f"Permission denied accessing directory '{subdirectory}'."}

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", args, summary)
        return summary

    @tool
    def inspect_data_tool(n_rows: StrictInt = 5) -> dict:
        """Peeks at the first n_rows of session.dataframe and returns summary stats and row previews.

        Args:
            n_rows: Positive integer number of rows to preview. Defaults to 5.

        Returns:
            dict: Summary containing result_id, total_rows, total_columns, columns,
                n_rows_previewed, and preview list of row dicts, or an error description.
        """
        args = {"n_rows": int(n_rows)}

        if session.dataframe is None:
            return {"error": "session.dataframe is None. Please load a dataset first."}

        if n_rows <= 0:
            return {"error": f"Invalid n_rows: {n_rows}. Must be a positive integer."}

        total_rows, total_cols = session.dataframe.shape
        raw_cols = list(session.dataframe.columns)
        str_cols = [str(c) for c in raw_cols]

        # Check duplicate string column names
        if len(str_cols) != len(set(str_cols)):
            return {"error": "DataFrame contains duplicate column names when converted to strings."}

        k = min(int(n_rows), total_rows)

        preview_records = []
        for r_idx in range(k):
            row_dict = {}
            for c_idx, str_col in enumerate(str_cols):
                raw_val = session.dataframe.iat[r_idx, c_idx]
                row_dict[str_col] = _sanitize_for_json(raw_val)
            preview_records.append(row_dict)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "columns": str_cols,
            "n_rows_previewed": k,
            "preview": preview_records,
        }

        # Validate JSON serialization without allowing NaNs before returning
        json.dumps(summary, allow_nan=False)

        session.store_result("inspect_data_tool", args, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Checks session.dataframe['text'] for missing or empty text values.

        Returns:
            dict: Summary containing result_id, total_rows, and missing_count,
                or an error description.
        """
        args = {}

        if session.dataframe is None:
            return {"error": "session.dataframe is None. Please load a dataset first."}

        if "text" not in session.dataframe.columns:
            return {"error": "session.dataframe missing required 'text' column."}

        missing_count = 0
        for val in session.dataframe["text"]:
            if pd.isna(val) or str(val).strip() == "":
                missing_count += 1

        total_rows = int(len(session.dataframe))
        missing_count = int(missing_count)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "missing_count": missing_count,
        }

        session.store_result("check_missing_tool", args, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for duplicate rows across full columns in session.dataframe, and optionally drops them using keep=False.

        Args:
            drop: If True, drops all copies of duplicated rows using keep=False and updates session state. Defaults to False.

        Returns:
            dict: Summary containing result_id, total_rows, duplicate_count, rows_dropped, and remaining_rows, or an error description.
        """
        args = {"drop": bool(drop)}

        if session.dataframe is None:
            return {"error": "session.dataframe is None. Please load a dataset first."}

        df = session.dataframe

        # Preparation phase: compute everything in local variables under try-except
        try:
            total_rows = int(len(df))

            # Check duplicate rows across all columns
            dup_mask_first = df.duplicated(keep="first")
            duplicate_count = int(dup_mask_first.sum())

            dup_mask_all = df.duplicated(keep=False)
            rows_dropped_if_drop = int(dup_mask_all.sum())

            new_df = None
            new_labels = None
            new_categories = None

            if drop and rows_dropped_if_drop > 0:
                # 1. Compute new dataframe locally without mutating session
                new_df = df[~dup_mask_all].reset_index(drop=True)
                rows_dropped = rows_dropped_if_drop
                remaining_rows = int(len(new_df))

                # 2. Compute updated labels and categories locally
                keep_indices = np.where(~dup_mask_all)[0]
                if "category_name" in new_df.columns:
                    raw_cats = new_df["category_name"].values
                    new_labels = np.asarray(raw_cats)
                    valid_cats = [c for c in pd.Series(raw_cats).dropna().unique()]
                    new_categories = sorted(list(valid_cats))
                elif session.labels is not None and len(session.labels) == total_rows:
                    new_labels = np.asarray(session.labels)[keep_indices]
                    valid_cats = [c for c in pd.Series(new_labels).dropna().unique()]
                    new_categories = sorted(list(valid_cats))
                else:
                    new_labels = None
                    new_categories = []
            else:
                rows_dropped = 0
                remaining_rows = total_rows

        except Exception as e:
            return {"error": f"Failed during duplicate checking/removal preparation: {str(e)}"}

        # State mutation phase: only performed after all calculations succeed
        if drop and rows_dropped > 0 and new_df is not None:
            session.dataframe = new_df
            session.labels = new_labels
            session.categories = new_categories
            session.feature_matrix = None
            session.feature_names = None
            session.artifacts = {}
            session.pending_figure = None

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "duplicate_count": duplicate_count,
            "rows_dropped": rows_dropped,
            "remaining_rows": remaining_rows,
        }

        session.store_result("check_duplicates_tool", args, summary)
        return summary

    @tool
    def sample_data_tool(n: StrictInt, random_state: StrictInt = 42) -> dict:
        """Subsamples session.dataframe without replacement using pandas sample, updates working dataset and labels/categories, and clears downstream analysis state.

        Args:
            n: Positive integer number of rows to sample (between 1 and total rows).
            random_state: Integer random seed for reproducible sampling. Defaults to 42.

        Returns:
            dict: Summary containing result_id, total_rows, sampled_rows, sampled_indices, and random_state, or an error description.
        """
        args = {"n": int(n), "random_state": int(random_state)}

        if session.dataframe is None:
            return {"error": "session.dataframe is None. Please load a dataset first."}

        df = session.dataframe

        # Preparation phase: compute everything in local variables under try-except
        try:
            total_rows = int(len(df))

            if total_rows == 0:
                return {"error": "Cannot sample from an empty dataset."}

            if n <= 0 or n > total_rows:
                return {"error": f"Invalid sample size n: {n}. Must be between 1 and total rows ({total_rows})."}

            # 1. Reset index on temp dataframe to track original 0-based row positions
            temp_df = df.reset_index(drop=True)

            # 2. Sample n rows without replacement
            sampled_temp_df = temp_df.sample(n=int(n), random_state=int(random_state), replace=False)

            # 3. Extract 0-based sampled indices and build reset new DataFrame
            sampled_indices = [int(idx) for idx in sampled_temp_df.index.tolist()]
            new_df = sampled_temp_df.reset_index(drop=True)

            # 4. Compute updated labels and categories locally
            if "category_name" in new_df.columns:
                raw_cats = new_df["category_name"].values
                new_labels = np.asarray(raw_cats)
                valid_cats = [c for c in pd.Series(raw_cats).dropna().unique()]
                new_categories = sorted(list(valid_cats))
            elif session.labels is not None and len(session.labels) == total_rows:
                new_labels = np.asarray(session.labels)[sampled_indices]
                valid_cats = [c for c in pd.Series(new_labels).dropna().unique()]
                new_categories = sorted(list(valid_cats))
            else:
                new_labels = None
                new_categories = []

            sampled_rows = int(len(new_df))

        except Exception as e:
            return {"error": f"Failed during sampling preparation: {str(e)}"}

        # State mutation phase: update session state only after all calculations succeed
        session.dataframe = new_df
        session.labels = new_labels
        session.categories = new_categories
        session.feature_matrix = None
        session.feature_names = None
        session.artifacts = {}
        session.pending_figure = None

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "sampled_rows": sampled_rows,
            "sampled_indices": sampled_indices,
            "random_state": int(random_state),
        }

        session.store_result("sample_data_tool", args, summary)
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Computes summary statistics of document text lengths overall and per category, and builds a box plot.

        Returns:
            dict: Summary containing result_id, overall stats dict, and per_category stats dict,
                or an error description.
        """
        args = {}

        fig = None
        try:
            if session.dataframe is None:
                return {"error": "session.dataframe is None. Please load a dataset first."}

            df = session.dataframe

            if df.empty:
                return {"error": "session.dataframe is empty."}

            if "text" not in df.columns or "category_name" not in df.columns:
                return {"error": "session.dataframe missing required 'text' or 'category_name' column."}

            # Check for invalid non-string or missing text values
            for val in df["text"]:
                if not isinstance(val, str):
                    return {"error": "session.dataframe['text'] contains missing or non-string values."}

            # 1. Local copy with text_length computed via len
            temp_df = df.copy()
            temp_df["text_length"] = temp_df["text"].apply(len)

            def _clean_stats(series):
                raw_dict = series.to_dict()
                cleaned = {}
                for key in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
                    val = raw_dict.get(key, None)
                    if val is None or pd.isna(val) or (isinstance(val, (float, np.floating)) and (math.isnan(val) or math.isinf(val))):
                        cleaned[key] = None
                    elif key == "count":
                        cleaned[key] = int(val)
                    else:
                        cleaned[key] = float(val)
                return cleaned

            # 2. Compute overall and per_category describe statistics
            overall_series = temp_df["text_length"].describe()
            overall_stats = _clean_stats(overall_series)

            per_category_stats = {}
            for cat_name, group in temp_df.groupby("category_name"):
                group_series = group["text_length"].describe()
                per_category_stats[str(cat_name)] = _clean_stats(group_series)

            # 3. Plot boxplot
            fig, ax = plt.subplots()
            sns.boxplot(data=temp_df, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Text Length")

            # 4. Prepare payload without result_id and validate JSON
            payload = {
                "overall": overall_stats,
                "per_category": per_category_stats,
            }
            json.dumps(payload, allow_nan=False)

        except Exception as e:
            if fig is not None:
                plt.close(fig)
            return {"error": f"Failed during describe_data preparation: {str(e)}"}

        # Preparation complete: increment counter, update session, and store result
        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall": overall_stats,
            "per_category": per_category_stats,
        }

        session.pending_figure = fig
        session.store_result("describe_data_tool", args, summary)
        return summary

    return [tokenize_tool, list_files_tool, inspect_data_tool, check_missing_tool, check_duplicates_tool, sample_data_tool, describe_data_tool]
