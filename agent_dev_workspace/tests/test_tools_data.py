import os
import json
import math
from unittest.mock import Mock
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_data import make_tools as make_data_tools

def _setup_fixture_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def _snapshot_session_state(session):
    """Takes a deep/copied snapshot of session attributes for state non-mutation assertions."""
    return {
        "dataframe": session.dataframe.copy() if session.dataframe is not None else None,
        "feature_matrix": session.feature_matrix.copy() if session.feature_matrix is not None else None,
        "feature_names": list(session.feature_names) if session.feature_names is not None else None,
        "labels": session.labels.copy() if session.labels is not None else None,
        "categories": list(session.categories) if session.categories is not None else None,
    }

def _assert_session_state_unchanged(session, orig_state):
    """Verifies that dataframe, feature_matrix, feature_names, labels, and categories
    remained unchanged after running a read-only tool.
    """
    if orig_state["dataframe"] is None:
        assert session.dataframe is None
    else:
        assert_frame_equal(session.dataframe, orig_state["dataframe"])

    if orig_state["feature_matrix"] is None:
        assert session.feature_matrix is None
    else:
        assert session.feature_matrix.shape == orig_state["feature_matrix"].shape
        if hasattr(orig_state["feature_matrix"], "toarray"):
            np.testing.assert_array_equal(session.feature_matrix.toarray(), orig_state["feature_matrix"].toarray())
        else:
            np.testing.assert_array_equal(session.feature_matrix, orig_state["feature_matrix"])

    assert session.feature_names == orig_state["feature_names"]

    if orig_state["labels"] is None:
        assert session.labels is None
    else:
        np.testing.assert_array_equal(session.labels, orig_state["labels"])

    if orig_state["categories"] is None:
        assert session.categories is None
    else:
        assert session.categories == orig_state["categories"]


def test_tokenize_tool_known_answer_10_and_records():
    session = _setup_fixture_session()
    
    orig_df = session.dataframe.copy()
    
    tools = make_data_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})
    assert "error" not in res

    # 1. Known Answer #10 Verification
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    assert res["n_documents"] == 8
    assert res["total_tokens"] == 27
    assert res["token_preview"] == expected_tokens[:3]

    # Verify per-document unigrams in dataframe and artifacts
    assert session.dataframe["unigrams"].tolist() == expected_tokens
    assert session.artifacts["unigrams"] == expected_tokens

    # Verify DataFrame non-mutation (except unigrams addition)
    df_without_unigrams = session.dataframe.drop(columns=["unigrams"])
    assert_frame_equal(df_without_unigrams, orig_df)

    # 3. Check record storage & JSON serialization
    rid = res["result_id"]
    stored = session.results_store[rid]
    record = stored["record"]

    assert record.args == {}
    assert record.summary == res
    json.dumps(res, allow_nan=False)

def test_tokenize_tool_case_duplicates_and_nulls():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": [
            "Hello HELLO hello World",  # Case conversion & repeated words
            None,                        # None
            np.nan,                      # NaN
            pd.NA,                       # pd.NA
            "",                          # Empty string
            "   "                       # Pure whitespace
        ]
    })

    tools = make_data_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})
    assert "error" not in res

    expected = [
        ["hello", "hello", "hello", "world"],
        [],
        [],
        [],
        [],
        []
    ]

    assert res["total_tokens"] == 4
    assert session.dataframe["unigrams"].tolist() == expected
    assert session.artifacts["unigrams"] == expected

def test_tokenize_tool_missing_dataframe_and_columns():
    # Missing dataframe
    s1 = SessionState()
    t1 = next(t for t in make_data_tools(s1) if t.name == "tokenize_tool")
    assert "error" in t1.invoke({})

    # Missing 'text' column
    s2 = SessionState()
    s2.dataframe = pd.DataFrame({"other_col": ["a", "b"]})
    t2 = next(t for t in make_data_tools(s2) if t.name == "tokenize_tool")
    assert "error" in t2.invoke({})

def test_tokenize_tool_lookup_error_atomicity(monkeypatch):
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["doc one", "doc two"]})
    
    # Take complete snapshots before call
    orig_df = session.dataframe.copy()
    orig_artifacts = dict(session.artifacts)
    orig_results_store = dict(session.results_store)

    tools = make_data_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    import nltk
    call_count = 0

    def mock_tokenize(text_str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ["doc", "one"]
        elif call_count == 2:
            raise LookupError("Simulated missing NLTK punkt resource")
        return ["dummy"]

    monkeypatch.setattr(nltk, "word_tokenize", mock_tokenize)

    res = tokenize_tool.invoke({})
    
    # 2. Confirm call_count == 2
    assert call_count == 2
    assert "error" in res
    assert "NLTK tokenizer resource not found" in res["error"]

    # 3. Confirm dataframe, artifacts, and results_store are completely untouched
    assert_frame_equal(session.dataframe, orig_df)
    assert session.artifacts == orig_artifacts
    assert session.results_store == orig_results_store

def test_list_files_tool_known_answer_11_and_records():
    session = _setup_fixture_session()

    # Build DTM using build_dtm_tool to ensure session.feature_matrix exists
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_res = build_dtm_tool.invoke({})
    assert "error" not in dtm_res

    orig_df = session.dataframe.copy()
    orig_fm = session.feature_matrix.copy()
    orig_fn = list(session.feature_names) if session.feature_names is not None else None
    orig_labels = session.labels.copy() if session.labels is not None else None

    tools = make_data_tools(session)
    list_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_tool.invoke({"subdirectory": "newdataset"})
    assert "error" not in res
    assert res["subdirectory"] == "newdataset"

    # Known Answer #11 Check
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

    # Record storage & JSON serialization check
    rid = res["result_id"]
    stored = session.results_store[rid]
    record = stored["record"]

    assert record.args == {"subdirectory": "newdataset"}
    assert record.summary == res
    json.dumps(res, allow_nan=False)

    # State non-mutation check
    assert_frame_equal(session.dataframe, orig_df)
    assert session.feature_matrix.shape == orig_fm.shape
    np.testing.assert_array_equal(session.feature_matrix.toarray(), orig_fm.toarray())
    assert session.feature_names == orig_fn
    if orig_labels is not None:
        np.testing.assert_array_equal(session.labels, orig_labels)

def test_list_files_tool_isolation_sorting_and_root_location(tmp_path, monkeypatch):
    # Setup mock repo root in tmp_path with pyproject.toml
    (tmp_path / "pyproject.toml").write_text("# mock pyproject")
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "b_file.txt").write_text("b")
    (tmp_path / "a_file.txt").write_text("a")

    # Subdirectory with nested items (test non-recursive listing)
    sub = tmp_path / "a_dir"
    (sub / "sub_file.txt").write_text("sub")
    (sub / "sub_folder").mkdir()
    (sub / "sub_folder" / "nested_file.txt").write_text("nested")

    # Test from root directory
    monkeypatch.chdir(tmp_path)
    session1 = SessionState()
    t1 = next(t for t in make_data_tools(session1) if t.name == "list_files_tool")

    res_root = t1.invoke({"subdirectory": ""})
    res_dot = t1.invoke({"subdirectory": "."})

    assert res_root["directories"] == ["a_dir", "b_dir"]
    assert res_root["files"] == ["a_file.txt", "b_file.txt", "pyproject.toml"]
    assert res_root["directories"] == res_dot["directories"]
    assert res_root["files"] == res_dot["files"]

    # Test invoking from sub directory (should locate same repo root)
    sub_dir = tmp_path / "b_dir"
    monkeypatch.chdir(sub_dir)

    session2 = SessionState()
    t2 = next(t for t in make_data_tools(session2) if t.name == "list_files_tool")

    res_sub = t2.invoke({"subdirectory": "a_dir"})
    assert res_sub["directories"] == ["sub_folder"]
    assert res_sub["files"] == ["sub_file.txt"]  # non-recursive: nested_file.txt not listed

def test_list_files_tool_edge_cases_and_errors(tmp_path, monkeypatch):
    # Setup mock repo root
    (tmp_path / "pyproject.toml").write_text("# mock")
    (tmp_path / "empty_dir").mkdir()
    (tmp_path / "some_file.txt").write_text("hello")
    monkeypatch.chdir(tmp_path)

    session = SessionState()
    list_tool = next(t for t in make_data_tools(session) if t.name == "list_files_tool")

    # 1. Empty directory
    res_empty = list_tool.invoke({"subdirectory": "empty_dir"})
    assert "error" not in res_empty
    assert res_empty["directories"] == []
    assert res_empty["files"] == []

    # 2. Path does not exist
    res_nonexist = list_tool.invoke({"subdirectory": "non_existing_folder"})
    assert "error" in res_nonexist

    # 3. Path points to a file, not a directory
    res_file = list_tool.invoke({"subdirectory": "some_file.txt"})
    assert "error" in res_file

    # 4. Out of bounds (directory traversal)
    res_out = list_tool.invoke({"subdirectory": "../outside"})
    assert "error" in res_out

    # 5. PermissionError simulation using monkeypatch on os.scandir
    def mock_scandir(path):
        raise PermissionError("Simulated permission denied")

    monkeypatch.setattr(os, "scandir", mock_scandir)
    res_perm = list_tool.invoke({"subdirectory": "empty_dir"})
    assert "error" in res_perm
    assert "Permission denied" in res_perm["error"]

def test_list_files_tool_missing_repo_root(tmp_path, monkeypatch):
    # Ensure no pyproject.toml exists in tmp_path or any of its parents
    isolated_dir = tmp_path / "no_root_here" / "sub"
    isolated_dir.mkdir(parents=True)
    monkeypatch.chdir(isolated_dir)

    original_exists = Path.exists

    def mock_exists(self):
        if self.name == "pyproject.toml":
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", mock_exists)

    session = SessionState()
    list_tool = next(t for t in make_data_tools(session) if t.name == "list_files_tool")

    res = list_tool.invoke({"subdirectory": ""})
    assert "error" in res
    assert "Could not locate repository root" in res["error"]

def test_inspect_data_tool_custom_preview():
    session = SessionState()
    session.dataframe = pd.DataFrame(
        {
            "col_a": ["alpha", "beta", "gamma", "delta"],
            "col_b": [10, 20, 30, 40],
            "col_c": [1.1, 2.2, 3.3, 4.4],
        },
        index=[100, 101, 102, 103],
    )

    orig_state = _snapshot_session_state(session)
    orig_results_store = dict(session.results_store)

    tools = make_data_tools(session)
    inspect_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res = inspect_tool.invoke({"n_rows": 3})
    assert "error" not in res

    assert res["total_rows"] == 4
    assert res["total_columns"] == 3
    assert res["columns"] == ["col_a", "col_b", "col_c"]
    assert res["n_rows_previewed"] == 3

    expected_preview = [
        {"col_a": "alpha", "col_b": 10, "col_c": 1.1},
        {"col_a": "beta", "col_b": 20, "col_c": 2.2},
        {"col_a": "gamma", "col_b": 30, "col_c": 3.3},
    ]
    assert res["preview"] == expected_preview

    # Verify result storage (only newly added result_id, existing untouched)
    rid = res["result_id"]
    assert rid not in orig_results_store
    assert set(session.results_store.keys()) == set(orig_results_store.keys()) | {rid}
    for old_k, old_v in orig_results_store.items():
        assert session.results_store[old_k] == old_v

    stored = session.results_store[rid]
    assert stored["record"].args == {"n_rows": 3}
    assert stored["record"].summary == res
    json.dumps(res, allow_nan=False)

    # State protection
    _assert_session_state_unchanged(session, orig_state)


def test_inspect_data_tool_defaults_and_truncation():
    session = _setup_fixture_session()

    # Build DTM using build_dtm_tool to ensure non-empty analysis state exists
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_res = build_dtm_tool.invoke({})
    assert "error" not in dtm_res

    orig_state = _snapshot_session_state(session)

    tools = make_data_tools(session)
    inspect_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # 1. Default n_rows=5 with non-empty results_store snapshot check
    store_before_default = dict(session.results_store)
    res_default = inspect_tool.invoke({})
    assert "error" not in res_default
    assert res_default["total_rows"] == 8
    assert res_default["n_rows_previewed"] == 5
    assert len(res_default["preview"]) == 5

    rid_default = res_default["result_id"]
    assert rid_default not in store_before_default
    assert set(session.results_store.keys()) == set(store_before_default.keys()) | {rid_default}
    for old_k, old_v in store_before_default.items():
        assert session.results_store[old_k] == old_v

    # 2. n_rows larger than total_rows (10 > 8) with non-empty results_store snapshot check
    store_before_large = dict(session.results_store)
    res_large = inspect_tool.invoke({"n_rows": 10})
    assert "error" not in res_large
    assert res_large["total_rows"] == 8
    assert res_large["n_rows_previewed"] == 8
    assert len(res_large["preview"]) == 8

    rid_large = res_large["result_id"]
    assert rid_large not in store_before_large
    assert set(session.results_store.keys()) == set(store_before_large.keys()) | {rid_large}
    for old_k, old_v in store_before_large.items():
        assert session.results_store[old_k] == old_v

    # State protection
    _assert_session_state_unchanged(session, orig_state)


def test_inspect_data_tool_missing_and_empty_dataframe():
    # Case 1: Unloaded session (session.dataframe is None)
    s1 = SessionState()
    orig_s1_store = dict(s1.results_store)
    t1 = next(t for t in make_data_tools(s1) if t.name == "inspect_data_tool")
    res1 = t1.invoke({"n_rows": 5})
    assert "error" in res1
    assert "session.dataframe is None" in res1["error"]
    assert s1.results_store == orig_s1_store

    # Case 2: Empty dataframe (0 rows)
    s2 = SessionState()
    s2.dataframe = pd.DataFrame(columns=["a", "b"])
    orig_s2 = _snapshot_session_state(s2)
    t2 = next(t for t in make_data_tools(s2) if t.name == "inspect_data_tool")
    res2 = t2.invoke({"n_rows": 5})
    assert "error" not in res2
    assert res2["total_rows"] == 0
    assert res2["total_columns"] == 2
    assert res2["n_rows_previewed"] == 0
    assert res2["preview"] == []
    _assert_session_state_unchanged(s2, orig_s2)


def test_inspect_data_tool_strict_int_validation():
    session = _setup_fixture_session()
    orig_state = _snapshot_session_state(session)
    orig_results_store = dict(session.results_store)

    tools = make_data_tools(session)
    inspect_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Invalid n_rows <= 0 handled inside tool logic
    res_zero = inspect_tool.invoke({"n_rows": 0})
    assert "error" in res_zero
    assert "Invalid n_rows" in res_zero["error"]

    res_neg = inspect_tool.invoke({"n_rows": -5})
    assert "error" in res_neg
    assert "Invalid n_rows" in res_neg["error"]

    # StrictInt validation failure via Pydantic (float, bool, str)
    for invalid_val in [3.5, 3.0, True, False, "3"]:
        with pytest.raises(ValidationError):
            inspect_tool.invoke({"n_rows": invalid_val})

    # Results store remains unchanged after validation failures / error responses
    assert session.results_store == orig_results_store
    _assert_session_state_unchanged(session, orig_state)


def test_inspect_data_tool_json_types_and_large_integers():
    session = SessionState()
    large_int = 2**63 - 1
    session.dataframe = pd.DataFrame(
        {
            "big_int": [large_int],
            "float_val": [np.float64(3.14159)],
            "pos_inf": [float("inf")],
            "neg_inf": [float("-inf")],
            "nan_val": [np.nan],
            "none_val": [None],
            "pd_na": [pd.NA],
            "nat_val": [pd.NaT],
            "bool_val": [np.bool_(True)],
            "dt_val": [pd.Timestamp("2026-03-31 12:00:00")],
            "nested_list": [[1, (2, 3)]],
            "nested_dict": [{"a": np.int64(10)}],
            "0d_array": [np.array(42)],
            "2d_array": [np.array([[1, 2], [3, 4]])],
        }
    )

    orig_state = _snapshot_session_state(session)

    tools = make_data_tools(session)
    inspect_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res = inspect_tool.invoke({"n_rows": 1})
    assert "error" not in res

    row = res["preview"][0]
    assert row["big_int"] == large_int
    assert isinstance(row["big_int"], int)
    assert row["float_val"] == pytest.approx(3.14159)
    assert row["pos_inf"] == "Infinity"
    assert row["neg_inf"] == "-Infinity"
    assert row["nan_val"] is None
    assert row["none_val"] is None
    assert row["pd_na"] is None
    assert row["nat_val"] is None
    assert row["bool_val"] is True
    assert row["dt_val"] == "2026-03-31T12:00:00"
    assert row["nested_list"] == [1, [2, 3]]
    assert row["nested_dict"] == {"a": 10}
    assert row["0d_array"] == 42
    assert row["2d_array"] == [[1, 2], [3, 4]]

    # Confirm strict JSON serializability without NaNs
    json.dumps(res, allow_nan=False)
    _assert_session_state_unchanged(session, orig_state)


def test_inspect_data_tool_column_name_string_conversion_and_collisions():
    # Case 1: Non-string column names convert cleanly to strings
    s1 = SessionState()
    s1.dataframe = pd.DataFrame([[1, 2]], columns=[100, 200])
    orig_s1 = _snapshot_session_state(s1)
    t1 = next(t for t in make_data_tools(s1) if t.name == "inspect_data_tool")
    res1 = t1.invoke({"n_rows": 1})
    assert "error" not in res1
    assert res1["columns"] == ["100", "200"]
    assert res1["preview"][0] == {"100": 1, "200": 2}
    _assert_session_state_unchanged(s1, orig_s1)

    # Case 2: Column name collisions after string conversion (int 100 vs str "100")
    s2 = SessionState()
    s2.dataframe = pd.DataFrame([[1, 2]], columns=[100, "100"])
    orig_s2 = _snapshot_session_state(s2)
    orig_s2_store = dict(s2.results_store)

    t2 = next(t for t in make_data_tools(s2) if t.name == "inspect_data_tool")
    res2 = t2.invoke({"n_rows": 1})
    assert "error" in res2
    assert "duplicate column names" in res2["error"]
    assert s2.results_store == orig_s2_store
    _assert_session_state_unchanged(s2, orig_s2)


def test_check_missing_tool_known_answer_2_and_records():
    session = _setup_fixture_session()

    # Build DTM to populate non-empty analysis state (feature_matrix, feature_names, etc.)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_res = build_dtm_tool.invoke({})
    assert "error" not in dtm_res

    orig_state = _snapshot_session_state(session)
    store_before = dict(session.results_store)

    tools = make_data_tools(session)
    check_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_tool.invoke({})
    assert "error" not in res

    # 1. Known Answer #2 Verification
    assert res["total_rows"] == 8
    assert res["missing_count"] == 1
    assert isinstance(res["total_rows"], int)
    assert type(res["total_rows"]) is int
    assert isinstance(res["missing_count"], int)
    assert type(res["missing_count"]) is int

    # Result storage check
    rid = res["result_id"]
    assert rid not in store_before
    assert set(session.results_store.keys()) == set(store_before.keys()) | {rid}
    for old_k, old_v in store_before.items():
        assert session.results_store[old_k] == old_v

    stored = session.results_store[rid]
    assert stored["record"].args == {}
    assert stored["record"].summary == res
    json.dumps(res, allow_nan=False)

    # State non-mutation check
    _assert_session_state_unchanged(session, orig_state)


def test_check_missing_tool_custom_text_values():
    session = SessionState()
    session.dataframe = pd.DataFrame(
        {
            "text": [
                None,
                np.nan,
                pd.NA,
                "",
                " \t\n",
                "hello",
            ]
        }
    )

    orig_state = _snapshot_session_state(session)
    orig_store = dict(session.results_store)

    tools = make_data_tools(session)
    check_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_tool.invoke({})
    assert "error" not in res
    assert res["total_rows"] == 6
    assert res["missing_count"] == 5

    # Storage check
    rid = res["result_id"]
    assert rid not in orig_store
    assert set(session.results_store.keys()) == set(orig_store.keys()) | {rid}

    _assert_session_state_unchanged(session, orig_state)


def test_check_missing_tool_empty_dataframe_and_errors():
    # Case 1: Empty dataframe with 'text' column
    s1 = SessionState()
    s1.dataframe = pd.DataFrame(columns=["text", "label"])
    orig_s1_state = _snapshot_session_state(s1)

    t1 = next(t for t in make_data_tools(s1) if t.name == "check_missing_tool")
    res1 = t1.invoke({})
    assert "error" not in res1
    assert res1["total_rows"] == 0
    assert res1["missing_count"] == 0
    _assert_session_state_unchanged(s1, orig_s1_state)

    # Case 2: Unloaded session (session.dataframe is None)
    s2 = SessionState()
    orig_s2_store = dict(s2.results_store)
    t2 = next(t for t in make_data_tools(s2) if t.name == "check_missing_tool")
    res2 = t2.invoke({})
    assert "error" in res2
    assert "session.dataframe is None" in res2["error"]
    assert s2.results_store == orig_s2_store

    # Case 3: Missing 'text' column
    s3 = SessionState()
    s3.dataframe = pd.DataFrame({"other_col": ["a", "b"]})
    orig_s3_state = _snapshot_session_state(s3)
    orig_s3_store = dict(s3.results_store)

    t3 = next(t for t in make_data_tools(s3) if t.name == "check_missing_tool")
    res3 = t3.invoke({})
    assert "error" in res3
    assert "missing required 'text' column" in res3["error"]
    assert s3.results_store == orig_s3_store
    _assert_session_state_unchanged(s3, orig_s3_state)


def test_check_duplicates_tool_known_answer_3_inspection_only():
    session = _setup_fixture_session()

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_res = build_dtm_tool.invoke({})
    assert "error" not in dtm_res

    # Specify test artifacts, pending_figure, and granted_tools
    session.artifacts = {"art_key": "art_val"}
    session.pending_figure = "fig_mark"
    session.granted_tools = {"check_duplicates_tool", "build_dtm_tool"}

    orig_state = _snapshot_session_state(session)
    orig_artifacts = dict(session.artifacts)
    orig_pending_figure = session.pending_figure
    orig_granted_tools = set(session.granted_tools)
    store_before = dict(session.results_store)

    tools = make_data_tools(session)
    check_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    res = check_tool.invoke({"drop": False})
    assert "error" not in res

    # 1. Known Answer #3 Inspection check
    assert res["total_rows"] == 8
    assert res["duplicate_count"] == 1
    assert res["rows_dropped"] == 0
    assert res["remaining_rows"] == 8

    # Native int verification
    for field in ["total_rows", "duplicate_count", "rows_dropped", "remaining_rows"]:
        assert isinstance(res[field], int)
        assert type(res[field]) is int

    # Results store & JSON verification
    rid = res["result_id"]
    assert rid not in store_before
    assert set(session.results_store.keys()) == set(store_before.keys()) | {rid}
    for old_k, old_v in store_before.items():
        assert session.results_store[old_k] == old_v

    stored = session.results_store[rid]
    assert stored["record"].args == {"drop": False}
    assert stored["record"].summary == res
    json.dumps(res, allow_nan=False)

    # State, artifacts, pending_figure & granted_tools non-mutation check
    _assert_session_state_unchanged(session, orig_state)
    assert session.artifacts == orig_artifacts
    assert session.pending_figure == orig_pending_figure
    assert set(session.granted_tools) == orig_granted_tools


def test_check_duplicates_tool_known_answer_3_drop_true_and_reset_state():
    session = _setup_fixture_session()

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_res = build_dtm_tool.invoke({})
    assert "error" not in dtm_res

    # Specify test artifacts, pending_figure mark, and non-empty granted_tools
    session.artifacts = {"test_art": 123}
    session.pending_figure = "test_fig_mark"
    session.granted_tools = {"check_duplicates_tool", "build_dtm_tool"}

    orig_df = session.dataframe.copy()
    orig_granted_tools = set(session.granted_tools)
    store_before = dict(session.results_store)

    tools = make_data_tools(session)
    check_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    res = check_tool.invoke({"drop": True})
    assert "error" not in res

    # 1. Known Answer #3 Drop=True check (keep=False drops both duplicate rows 5 and 6)
    assert res["total_rows"] == 8
    assert res["duplicate_count"] == 1
    assert res["rows_dropped"] == 2
    assert res["remaining_rows"] == 6

    # Native int verification
    for field in ["total_rows", "duplicate_count", "rows_dropped", "remaining_rows"]:
        assert isinstance(res[field], int)
        assert type(res[field]) is int

    # Retained rows check using original DataFrame slice [0, 1, 2, 3, 4, 7] with reset index
    expected_df = orig_df.iloc[[0, 1, 2, 3, 4, 7]].reset_index(drop=True)
    assert_frame_equal(session.dataframe, expected_df)

    # Labels and categories alignment based on retained category_name values
    retained_cats = expected_df["category_name"]
    expected_categories = sorted(retained_cats.dropna().unique().tolist())
    expected_labels = retained_cats.values

    assert session.labels is not None
    np.testing.assert_array_equal(session.labels, expected_labels)
    assert session.categories == expected_categories

    # Cleared analysis state check
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.artifacts == {}
    assert session.pending_figure is None

    # Non-empty Granted tools & Results store retention check
    assert set(session.granted_tools) == orig_granted_tools
    rid = res["result_id"]
    assert rid not in store_before
    assert set(session.results_store.keys()) == set(store_before.keys()) | {rid}
    for old_k, old_v in store_before.items():
        assert session.results_store[old_k] == old_v

    stored = session.results_store[rid]
    assert stored["record"].args == {"drop": True}
    assert stored["record"].summary == res
    json.dumps(res, allow_nan=False)


def test_check_duplicates_tool_label_fallback_mechanisms():
    # Case A: No category_name column (fallback to original session.labels filtering)
    s1 = SessionState()
    s1.dataframe = pd.DataFrame(
        {
            "text": ["dup", "dup", "unique1", "unique2"],
            "category": [0, 0, 1, 0],
        }
    )
    s1.labels = np.array(["cat0", "cat0", "cat1", "cat0"])
    s1.categories = ["cat0", "cat1"]

    tools1 = make_data_tools(s1)
    t1 = next(t for t in tools1 if t.name == "check_duplicates_tool")
    res1 = t1.invoke({"drop": True})

    assert "error" not in res1
    assert res1["rows_dropped"] == 2
    assert len(s1.dataframe) == 2
    np.testing.assert_array_equal(s1.labels, np.array(["cat1", "cat0"]))
    assert s1.categories == ["cat0", "cat1"]

    # Case B: No label sources at all (session.labels=None, no category_name/category column)
    s2 = SessionState()
    s2.dataframe = pd.DataFrame(
        {
            "text": ["dup", "dup", "unique1", "unique2"],
        }
    )
    s2.labels = None
    s2.categories = []

    tools2 = make_data_tools(s2)
    t2 = next(t for t in tools2 if t.name == "check_duplicates_tool")
    res2 = t2.invoke({"drop": True})

    assert "error" not in res2
    assert res2["rows_dropped"] == 2
    assert len(s2.dataframe) == 2
    assert s2.labels is None
    assert s2.categories == []


def test_check_duplicates_tool_boundaries_and_zero_drop_state_preservation():
    # Boundary 1: No duplicates
    s1 = SessionState()
    s1.dataframe = pd.DataFrame({"text": ["doc1", "doc2", "doc3"], "category_name": ["catA", "catB", "catA"]})
    s1.feature_matrix = np.ones((3, 2))
    s1.feature_names = ["term1", "term2"]
    s1.artifacts = {"art": 1}
    s1.pending_figure = "fig"
    s1.granted_tools = {"check_duplicates_tool"}

    orig_s1 = _snapshot_session_state(s1)
    orig_s1_art = dict(s1.artifacts)
    orig_s1_fig = s1.pending_figure
    orig_s1_granted = set(s1.granted_tools)

    t1 = next(t for t in make_data_tools(s1) if t.name == "check_duplicates_tool")
    res1 = t1.invoke({"drop": True})

    assert "error" not in res1
    assert res1["total_rows"] == 3
    assert res1["duplicate_count"] == 0
    assert res1["rows_dropped"] == 0
    assert res1["remaining_rows"] == 3
    _assert_session_state_unchanged(s1, orig_s1)
    assert s1.artifacts == orig_s1_art
    assert s1.pending_figure == orig_s1_fig
    assert set(s1.granted_tools) == orig_s1_granted

    # Boundary 2: Empty DataFrame (0 rows)
    s2 = SessionState()
    s2.dataframe = pd.DataFrame(columns=["text", "category_name"])
    s2.artifacts = {"art": 2}
    s2.pending_figure = "fig2"
    s2.granted_tools = {"check_duplicates_tool"}

    orig_s2 = _snapshot_session_state(s2)
    orig_s2_art = dict(s2.artifacts)
    orig_s2_fig = s2.pending_figure
    orig_s2_granted = set(s2.granted_tools)

    t2 = next(t for t in make_data_tools(s2) if t.name == "check_duplicates_tool")
    res2 = t2.invoke({"drop": True})

    assert "error" not in res2
    assert res2["total_rows"] == 0
    assert res2["duplicate_count"] == 0
    assert res2["rows_dropped"] == 0
    assert res2["remaining_rows"] == 0
    _assert_session_state_unchanged(s2, orig_s2)
    assert s2.artifacts == orig_s2_art
    assert s2.pending_figure == orig_s2_fig
    assert set(s2.granted_tools) == orig_s2_granted

    # Boundary 3: Three identical rows (extra copies = 2, all dropped = 3, 0 remaining rows)
    s3 = SessionState()
    s3.dataframe = pd.DataFrame({"text": ["same", "same", "same"], "category_name": ["catA", "catA", "catA"]})
    s3.granted_tools = {"check_duplicates_tool"}
    orig_s3_granted = set(s3.granted_tools)

    t3 = next(t for t in make_data_tools(s3) if t.name == "check_duplicates_tool")
    res3 = t3.invoke({"drop": True})

    assert "error" not in res3
    assert res3["total_rows"] == 3
    assert res3["duplicate_count"] == 2
    assert res3["rows_dropped"] == 3
    assert res3["remaining_rows"] == 0
    assert len(s3.dataframe) == 0
    assert s3.labels is not None
    assert len(s3.labels) == 0
    assert s3.categories == []
    assert set(s3.granted_tools) == orig_s3_granted


def test_check_duplicates_tool_error_atomicity(monkeypatch):
    # Error Case 1: session.dataframe is None
    s1 = SessionState()
    s1.artifacts = {"art1": 1}
    s1.pending_figure = "fig1"
    s1.granted_tools = {"check_duplicates_tool"}

    orig_s1_store = dict(s1.results_store)
    orig_s1_art = dict(s1.artifacts)
    orig_s1_fig = s1.pending_figure
    orig_s1_granted = set(s1.granted_tools)

    t1 = next(t for t in make_data_tools(s1) if t.name == "check_duplicates_tool")
    res1 = t1.invoke({"drop": False})

    assert "error" in res1
    assert "session.dataframe is None" in res1["error"]
    assert s1.results_store == orig_s1_store
    assert s1.artifacts == orig_s1_art
    assert s1.pending_figure == orig_s1_fig
    assert set(s1.granted_tools) == orig_s1_granted

    # Error Case 2: DataFrame.duplicated raises exception (drop=False mode)
    s2 = _setup_fixture_session()
    s2.artifacts = {"art2": 2}
    s2.pending_figure = "fig2"
    s2.granted_tools = {"check_duplicates_tool"}

    orig_s2_state = _snapshot_session_state(s2)
    orig_s2_store = dict(s2.results_store)
    orig_s2_art = dict(s2.artifacts)
    orig_s2_fig = s2.pending_figure
    orig_s2_granted = set(s2.granted_tools)

    def mock_duplicated_fail(*args, **kwargs):
        raise RuntimeError("Simulated duplicated failure")

    monkeypatch.setattr(pd.DataFrame, "duplicated", mock_duplicated_fail)

    t2 = next(t for t in make_data_tools(s2) if t.name == "check_duplicates_tool")
    res2 = t2.invoke({"drop": False})

    assert "error" in res2
    assert "Simulated duplicated failure" in res2["error"]
    _assert_session_state_unchanged(s2, orig_s2_state)
    assert s2.results_store == orig_s2_store
    assert s2.artifacts == orig_s2_art
    assert s2.pending_figure == orig_s2_fig
    assert set(s2.granted_tools) == orig_s2_granted

    # Error Case 3: DataFrame.duplicated raises exception (drop=True mode)
    s3 = _setup_fixture_session()
    s3.artifacts = {"art3": 3}
    s3.pending_figure = "fig3"
    s3.granted_tools = {"check_duplicates_tool"}

    orig_s3_state = _snapshot_session_state(s3)
    orig_s3_store = dict(s3.results_store)
    orig_s3_art = dict(s3.artifacts)
    orig_s3_fig = s3.pending_figure
    orig_s3_granted = set(s3.granted_tools)

    t3 = next(t for t in make_data_tools(s3) if t.name == "check_duplicates_tool")
    res3 = t3.invoke({"drop": True})

    assert "error" in res3
    assert "Simulated duplicated failure" in res3["error"]
    _assert_session_state_unchanged(s3, orig_s3_state)
    assert s3.results_store == orig_s3_store
    assert s3.artifacts == orig_s3_art
    assert s3.pending_figure == orig_s3_fig
    assert set(s3.granted_tools) == orig_s3_granted

    # Error Case 4: Label factorize / derivation failure
    # First un-mock duplicated
    monkeypatch.undo()

    # Build fixture session & non-empty analysis state (DTM)
    s4 = _setup_fixture_session()
    dtm_tools = make_dtm_tools(s4)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_res = build_dtm_tool.invoke({})
    assert "error" not in dtm_res

    # Specify test artifacts, pending_figure mark, and granted_tools before mocking unique
    s4.artifacts = {"art4": 4}
    s4.pending_figure = "fig4"
    s4.granted_tools = {"check_duplicates_tool", "build_dtm_tool"}

    orig_s4_state = _snapshot_session_state(s4)
    orig_s4_store = dict(s4.results_store)
    orig_s4_art = dict(s4.artifacts)
    orig_s4_fig = s4.pending_figure
    orig_s4_granted = set(s4.granted_tools)

    # Mock pd.Series.unique to raise exception during label factorizing
    def mock_unique_fail(self):
        raise ValueError("Simulated label derivation failure")

    monkeypatch.setattr(pd.Series, "unique", mock_unique_fail)

    t4 = next(t for t in make_data_tools(s4) if t.name == "check_duplicates_tool")
    res4 = t4.invoke({"drop": True})

    assert "error" in res4
    assert "Simulated label derivation failure" in res4["error"]
    _assert_session_state_unchanged(s4, orig_s4_state)
    assert s4.results_store == orig_s4_store
    assert s4.artifacts == orig_s4_art
    assert s4.pending_figure == orig_s4_fig
    assert set(s4.granted_tools) == orig_s4_granted


def test_sample_data_tool_known_answer_13_and_records():
    session = _setup_fixture_session()

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_res = build_dtm_tool.invoke({})
    assert "error" not in dtm_res

    session.artifacts = {"test_art": 123}
    session.pending_figure = "test_fig_mark"
    session.granted_tools = {"sample_data_tool", "build_dtm_tool"}
    session.raw_dataset = {"raw_key": "raw_val"}

    orig_df = session.dataframe.copy()
    orig_granted_tools = set(session.granted_tools)
    orig_raw_dataset = dict(session.raw_dataset) if isinstance(session.raw_dataset, dict) else session.raw_dataset
    store_before = dict(session.results_store)

    tools = make_data_tools(session)
    sample_tool = next(t for t in tools if t.name == "sample_data_tool")

    res = sample_tool.invoke({"n": 4})
    assert "error" not in res

    # Known Answer #13 Verification (n=4, default random_state=42 => indices [1, 5, 0, 7])
    assert res["total_rows"] == 8
    assert res["sampled_rows"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert res["random_state"] == 42

    # Native int verification for result fields and sampled_indices elements
    for field in ["total_rows", "sampled_rows", "random_state"]:
        assert isinstance(res[field], int)
        assert type(res[field]) is int
    for idx_val in res["sampled_indices"]:
        assert isinstance(idx_val, int)
        assert type(idx_val) is int

    # Retained DataFrame check using iloc[[1, 5, 0, 7]].reset_index(drop=True)
    expected_df = orig_df.iloc[[1, 5, 0, 7]].reset_index(drop=True)
    assert_frame_equal(session.dataframe, expected_df)

    # Labels and categories alignment verification
    retained_cats = expected_df["category_name"]
    expected_categories = sorted(retained_cats.dropna().unique().tolist())
    expected_labels = retained_cats.values

    assert session.labels is not None
    np.testing.assert_array_equal(session.labels, expected_labels)
    assert session.categories == expected_categories

    # Cleared analysis state check
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert session.artifacts == {}
    assert session.pending_figure is None

    # Granted tools, raw_dataset & Results store retention check
    assert set(session.granted_tools) == orig_granted_tools
    assert session.raw_dataset == orig_raw_dataset
    rid = res["result_id"]
    assert rid not in store_before
    assert set(session.results_store.keys()) == set(store_before.keys()) | {rid}
    for old_k, old_v in store_before.items():
        assert session.results_store[old_k] == old_v

    stored = session.results_store[rid]
    assert stored["record"].args == {"n": 4, "random_state": 42}
    assert stored["record"].summary == res
    json.dumps(res, allow_nan=False)


def test_sample_data_tool_reproducibility_and_boundaries():
    # 1. Reproducibility using 2 independent sessions with same initial state
    s1 = _setup_fixture_session()
    s2 = _setup_fixture_session()

    t1 = next(t for t in make_data_tools(s1) if t.name == "sample_data_tool")
    t2 = next(t for t in make_data_tools(s2) if t.name == "sample_data_tool")

    res1 = t1.invoke({"n": 3, "random_state": 100})
    res2 = t2.invoke({"n": 3, "random_state": 100})

    assert res1["sampled_indices"] == res2["sampled_indices"]
    assert_frame_equal(s1.dataframe, s2.dataframe)

    # 2. Boundary: n = 1
    s_min = _setup_fixture_session()
    t_min = next(t for t in make_data_tools(s_min) if t.name == "sample_data_tool")
    res_min = t_min.invoke({"n": 1, "random_state": 42})
    assert "error" not in res_min
    assert res_min["sampled_rows"] == 1
    assert len(res_min["sampled_indices"]) == 1
    assert len(s_min.dataframe) == 1

    # 3. Boundary: n = total_rows (n = 8) with DTM check and label/category sync check
    s_all = _setup_fixture_session()
    dtm_tools = make_dtm_tools(s_all)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_res = build_dtm_tool.invoke({})
    assert "error" not in dtm_res

    s_all.artifacts = {"art": 1}
    s_all.pending_figure = "fig"
    s_all.granted_tools = {"sample_data_tool"}
    s_all.raw_dataset = {"raw": "data"}

    orig_s_all_granted = set(s_all.granted_tools)
    orig_s_all_raw = dict(s_all.raw_dataset)
    orig_all_df = s_all.dataframe.copy()

    t_all = next(t for t in make_data_tools(s_all) if t.name == "sample_data_tool")
    res_all = t_all.invoke({"n": 8, "random_state": 42})
    assert "error" not in res_all
    assert res_all["sampled_rows"] == 8
    assert set(res_all["sampled_indices"]) == set(range(8))

    # Order shuffled by sample; assert DataFrame, labels & categories equal re-ordered iloc
    expected_all_df = orig_all_df.iloc[res_all["sampled_indices"]].reset_index(drop=True)
    assert_frame_equal(s_all.dataframe, expected_all_df)

    expected_all_cats = sorted(expected_all_df["category_name"].dropna().unique().tolist())
    expected_all_labels = expected_all_df["category_name"].values
    np.testing.assert_array_equal(s_all.labels, expected_all_labels)
    assert s_all.categories == expected_all_cats

    # Cleared analysis state & preserved granted_tools/raw_dataset check
    assert s_all.feature_matrix is None
    assert s_all.feature_names is None
    assert s_all.artifacts == {}
    assert s_all.pending_figure is None
    assert set(s_all.granted_tools) == orig_s_all_granted
    assert s_all.raw_dataset == orig_s_all_raw


def test_sample_data_tool_index_tracking_and_label_fallbacks():
    # Case A: Non-continuous and duplicate index DataFrame (position tracking check & labels fallback)
    s1 = SessionState()
    s1.dataframe = pd.DataFrame(
        {
            "text": ["doc0", "doc1", "doc2", "doc3", "doc4"],
            "category": [10, 20, 10, 30, 20],
        },
        index=[100, 100, 200, 300, 200],  # duplicate & non-0-based index
    )
    s1.labels = np.array(["c10", "c20", "c10", "c30", "c20"])
    s1.categories = ["c10", "c20", "c30"]

    orig_s1_df = s1.dataframe.copy()
    orig_s1_labels = np.asarray(s1.labels).copy()

    t1 = next(t for t in make_data_tools(s1) if t.name == "sample_data_tool")
    res1 = t1.invoke({"n": 3, "random_state": 42})

    assert "error" not in res1
    assert res1["total_rows"] == 5
    assert res1["sampled_rows"] == 3
    # sampled_indices must reflect 0-based positions (0..4), not index values
    sampled_pos = res1["sampled_indices"]
    assert all(0 <= idx < 5 for idx in sampled_pos)

    # Verify DataFrame using orig_s1_df.iloc[sampled_pos].reset_index(drop=True)
    expected_df1 = orig_s1_df.iloc[sampled_pos].reset_index(drop=True)
    assert_frame_equal(s1.dataframe, expected_df1)

    # Verify labels using orig_s1_labels[sampled_pos]
    expected_labels1 = orig_s1_labels[sampled_pos]
    np.testing.assert_array_equal(s1.labels, expected_labels1)
    expected_categories1 = sorted(pd.Series(expected_labels1).dropna().unique().tolist())
    assert s1.categories == expected_categories1

    # Case B: No label sources at all (session.labels=None, no category_name/category column)
    s2 = SessionState()
    s2.dataframe = pd.DataFrame({"text": ["a", "b", "c", "d"]})
    s2.labels = None
    s2.categories = []

    t2 = next(t for t in make_data_tools(s2) if t.name == "sample_data_tool")
    res2 = t2.invoke({"n": 2, "random_state": 42})

    assert "error" not in res2
    assert s2.labels is None
    assert s2.categories == []


def test_sample_data_tool_input_errors_and_strictint_validation():
    # 1. session.dataframe is None
    s1 = SessionState()
    s1.raw_dataset = {"raw": 1}
    orig_store1 = dict(s1.results_store)
    orig_raw1 = dict(s1.raw_dataset)

    t1 = next(t for t in make_data_tools(s1) if t.name == "sample_data_tool")
    res1 = t1.invoke({"n": 2})
    assert "error" in res1
    assert "session.dataframe is None" in res1["error"]
    assert s1.results_store == orig_store1
    assert s1.raw_dataset == orig_raw1

    # 2. Empty DataFrame (0 rows)
    s2 = SessionState()
    s2.dataframe = pd.DataFrame(columns=["text", "category_name"])
    s2.raw_dataset = {"raw": 2}
    orig_s2_state = _snapshot_session_state(s2)
    orig_store2 = dict(s2.results_store)
    orig_raw2 = dict(s2.raw_dataset)

    t2 = next(t for t in make_data_tools(s2) if t.name == "sample_data_tool")
    res2 = t2.invoke({"n": 1})
    assert "error" in res2
    assert "empty dataset" in res2["error"]
    _assert_session_state_unchanged(s2, orig_s2_state)
    assert s2.results_store == orig_store2
    assert s2.raw_dataset == orig_raw2

    # 3. Invalid n <= 0, n > total_rows, or invalid seed (random_state=-1)
    s3 = _setup_fixture_session()
    s3.raw_dataset = {"raw": 3}
    orig_s3_state = _snapshot_session_state(s3)
    orig_store3 = dict(s3.results_store)
    orig_raw3 = dict(s3.raw_dataset)

    t3 = next(t for t in make_data_tools(s3) if t.name == "sample_data_tool")

    res3_zero = t3.invoke({"n": 0})
    assert "error" in res3_zero
    assert "Invalid sample size n" in res3_zero["error"]

    res3_neg = t3.invoke({"n": -3})
    assert "error" in res3_neg
    assert "Invalid sample size n" in res3_neg["error"]

    res3_over = t3.invoke({"n": 100})
    assert "error" in res3_over
    assert "Invalid sample size n" in res3_over["error"]

    res3_bad_seed = t3.invoke({"n": 3, "random_state": -1})
    assert "error" in res3_bad_seed
    assert "Invalid sample size n" in res3_bad_seed["error"] or "random_state" in res3_bad_seed["error"] or "Failed during sampling" in res3_bad_seed["error"]

    _assert_session_state_unchanged(s3, orig_s3_state)
    assert s3.results_store == orig_store3
    assert s3.raw_dataset == orig_raw3

    # 4. StrictInt Pydantic validation failures for non-int, float, bool, or str
    for invalid_n in [2.5, True, False, "2"]:
        with pytest.raises(ValidationError):
            t3.invoke({"n": invalid_n})

    for invalid_seed in [42.5, True, False, "42"]:
        with pytest.raises(ValidationError):
            t3.invoke({"n": 3, "random_state": invalid_seed})

    _assert_session_state_unchanged(s3, orig_s3_state)
    assert s3.results_store == orig_store3
    assert s3.raw_dataset == orig_raw3


def test_sample_data_tool_preparation_error_atomicity():
    # Error Case 1: DataFrame.sample raises exception
    s1 = _setup_fixture_session()
    dtm_tools1 = make_dtm_tools(s1)
    build_dtm_tool1 = next(t for t in dtm_tools1 if t.name == "build_dtm_tool")
    dtm_res1 = build_dtm_tool1.invoke({})
    assert "error" not in dtm_res1

    s1.artifacts = {"art1": 1}
    s1.pending_figure = "fig1"
    s1.granted_tools = {"sample_data_tool", "build_dtm_tool"}
    s1.raw_dataset = {"raw1": 1}

    orig_s1_state = _snapshot_session_state(s1)
    orig_s1_store = dict(s1.results_store)
    orig_s1_art = dict(s1.artifacts)
    orig_s1_fig = s1.pending_figure
    orig_s1_granted = set(s1.granted_tools)
    orig_s1_raw = dict(s1.raw_dataset)

    tools1 = make_data_tools(s1)
    t1 = next(t for t in tools1 if t.name == "sample_data_tool")

    with pytest.MonkeyPatch.context() as mp:
        def mock_sample_fail(*args, **kwargs):
            raise RuntimeError("Simulated sample calculation failure")

        mp.setattr(pd.DataFrame, "sample", mock_sample_fail)
        res1 = t1.invoke({"n": 3})

        assert "error" in res1
        assert "Simulated sample calculation failure" in res1["error"]
        _assert_session_state_unchanged(s1, orig_s1_state)
        assert s1.results_store == orig_s1_store
        assert s1.artifacts == orig_s1_art
        assert s1.pending_figure == orig_s1_fig
        assert set(s1.granted_tools) == orig_s1_granted
        assert s1.raw_dataset == orig_s1_raw

    # Error Case 2: Series.unique raises exception during label derivation
    s2 = _setup_fixture_session()
    dtm_tools2 = make_dtm_tools(s2)
    build_dtm_tool2 = next(t for t in dtm_tools2 if t.name == "build_dtm_tool")
    dtm_res2 = build_dtm_tool2.invoke({})
    assert "error" not in dtm_res2

    s2.artifacts = {"art2": 2}
    s2.pending_figure = "fig2"
    s2.granted_tools = {"sample_data_tool", "build_dtm_tool"}
    s2.raw_dataset = {"raw2": 2}

    orig_s2_state = _snapshot_session_state(s2)
    orig_s2_store = dict(s2.results_store)
    orig_s2_art = dict(s2.artifacts)
    orig_s2_fig = s2.pending_figure
    orig_s2_granted = set(s2.granted_tools)
    orig_s2_raw = dict(s2.raw_dataset)

    tools2 = make_data_tools(s2)
    t2 = next(t for t in tools2 if t.name == "sample_data_tool")

    with pytest.MonkeyPatch.context() as mp:
        def mock_unique_fail(self):
            raise ValueError("Simulated label derivation failure in sample")

        mp.setattr(pd.Series, "unique", mock_unique_fail)
        res2 = t2.invoke({"n": 3})

        assert "error" in res2
        assert "Simulated label derivation failure in sample" in res2["error"]
        _assert_session_state_unchanged(s2, orig_s2_state)
        assert s2.results_store == orig_s2_store
        assert s2.artifacts == orig_s2_art
        assert s2.pending_figure == orig_s2_fig
        assert set(s2.granted_tools) == orig_s2_granted
        assert s2.raw_dataset == orig_s2_raw


def test_describe_data_tool_known_answer_18():
    session = _setup_fixture_session()
    tools = make_data_tools(session)
    describe_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_tool.invoke({})
    assert "error" not in res

    stat_keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    # Overall Known Answer #18
    expected_overall = [8, 19.875, 9.4330, 0, 17.75, 20.5, 25.25, 30]
    overall_dict = res["overall"]
    assert list(overall_dict.keys()) == stat_keys
    assert overall_dict["count"] == 8
    assert type(overall_dict["count"]) is int
    for idx, key in enumerate(stat_keys):
        if key == "count":
            continue
        val = overall_dict[key]
        assert type(val) is float
        assert val == pytest.approx(expected_overall[idx], abs=1e-4)

    # per_category Known Answer #18
    expected_catA = [4, 17.25, 12.5, 0, 12.75, 20.0, 24.5, 29]
    expected_catB = [4, 22.5, 5.7446, 18, 18.0, 21.0, 25.5, 30]

    assert set(res["per_category"].keys()) == {"catA", "catB"}

    try:
        catA_dict = res["per_category"]["catA"]
        assert list(catA_dict.keys()) == stat_keys
        assert catA_dict["count"] == 4
        assert type(catA_dict["count"]) is int
        for idx, key in enumerate(stat_keys):
            if key == "count":
                continue
            val = catA_dict[key]
            assert type(val) is float
            assert val == pytest.approx(expected_catA[idx], abs=1e-4)

        catB_dict = res["per_category"]["catB"]
        assert list(catB_dict.keys()) == stat_keys
        assert catB_dict["count"] == 4
        assert type(catB_dict["count"]) is int
        for idx, key in enumerate(stat_keys):
            if key == "count":
                continue
            val = catB_dict[key]
            assert type(val) is float
            assert val == pytest.approx(expected_catB[idx], abs=1e-4)

        # Confirm strict JSON serializability without NaNs
        json.dumps(res, allow_nan=False)

    finally:
        if session.pending_figure is not None:
            plt.close(session.pending_figure)


def test_describe_data_tool_success_state_figure_and_records():
    session = _setup_fixture_session()

    dtm_tools = make_dtm_tools(session)
    dtm_res = next(t for t in dtm_tools if t.name == "build_dtm_tool").invoke({})
    assert "error" not in dtm_res

    session.artifacts = {"art_key": "art_val"}
    session.raw_dataset = {"raw": 1}
    session.granted_tools = {"describe_data_tool", "build_dtm_tool"}

    orig_state = _snapshot_session_state(session)
    orig_art = dict(session.artifacts)
    orig_raw = dict(session.raw_dataset)
    orig_granted = set(session.granted_tools)
    store_before = dict(session.results_store)

    tools = make_data_tools(session)
    describe_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_tool.invoke({})
    assert "error" not in res

    try:
        # State non-mutation check (ensure no 'text_length' added to session.dataframe)
        _assert_session_state_unchanged(session, orig_state)
        assert "text_length" not in session.dataframe.columns
        assert session.artifacts == orig_art
        assert session.raw_dataset == orig_raw
        assert set(session.granted_tools) == orig_granted

        # Check pending_figure
        fig = session.pending_figure
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        xticklabels = [label.get_text() for label in fig.axes[0].get_xticklabels()]
        assert "catA" in xticklabels and "catB" in xticklabels

        # Results store & JSON check
        rid = res["result_id"]
        assert rid not in store_before
        assert set(session.results_store.keys()) == set(store_before.keys()) | {rid}
        for old_k, old_v in store_before.items():
            assert session.results_store[old_k] == old_v

        stored = session.results_store[rid]
        assert stored["record"].args == {}
        assert stored["record"].summary == res

    finally:
        if session.pending_figure is not None:
            plt.close(session.pending_figure)


def test_describe_data_tool_boundaries_nan_inf_and_json():
    # 1. Single row and empty string
    s1 = SessionState()
    s1.dataframe = pd.DataFrame({"text": [""], "category_name": ["catA"]})
    tools1 = make_data_tools(s1)
    t1 = next(t for t in tools1 if t.name == "describe_data_tool")

    res1 = t1.invoke({})
    assert "error" not in res1
    try:
        assert res1["overall"]["count"] == 1
        assert res1["overall"]["min"] == 0.0
        assert res1["overall"]["max"] == 0.0
        assert res1["overall"]["std"] is None
        assert res1["per_category"]["catA"]["std"] is None
        json.dumps(res1, allow_nan=False)
    finally:
        if s1.pending_figure is not None:
            plt.close(s1.pending_figure)

    # 2. Mock describe returning NaN and inf
    s2 = SessionState()
    s2.dataframe = pd.DataFrame({"text": ["a", "bb"], "category_name": ["catA", "catA"]})
    tools2 = make_data_tools(s2)
    t2 = next(t for t in tools2 if t.name == "describe_data_tool")

    mock_series = pd.Series({
        "count": np.nan,
        "mean": float("inf"),
        "std": float("-inf"),
        "min": np.nan,
        "25%": float("nan"),
        "50%": 1.5,
        "75%": float("inf"),
        "max": 2.0
    })

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd.Series, "describe", lambda self: mock_series)
        res2 = t2.invoke({})
        assert "error" not in res2
        try:
            expected_nan_stats = {
                "count": None,
                "mean": None,
                "std": None,
                "min": None,
                "25%": None,
                "50%": 1.5,
                "75%": None,
                "max": 2.0,
            }
            assert res2["overall"] == expected_nan_stats
            assert res2["per_category"]["catA"] == expected_nan_stats
            json.dumps(res2, allow_nan=False)
        finally:
            if s2.pending_figure is not None:
                plt.close(s2.pending_figure)


def test_describe_data_tool_input_validation_errors():
    # Case 1: Unloaded session
    s1 = SessionState()
    s1.artifacts = {"art": 1}
    s1.raw_dataset = {"raw": 1}
    s1.granted_tools = {"describe_data_tool"}
    s1.pending_figure = "old_fig"

    orig_s1_state = _snapshot_session_state(s1)
    orig_store1 = dict(s1.results_store)
    orig_art1 = dict(s1.artifacts)
    orig_raw1 = dict(s1.raw_dataset)
    orig_granted1 = set(s1.granted_tools)
    orig_fig1 = s1.pending_figure

    t1 = next(t for t in make_data_tools(s1) if t.name == "describe_data_tool")
    res1 = t1.invoke({})
    assert "error" in res1
    assert "session.dataframe is None" in res1["error"]
    _assert_session_state_unchanged(s1, orig_s1_state)
    assert s1.results_store == orig_store1
    assert s1.artifacts == orig_art1
    assert s1.raw_dataset == orig_raw1
    assert set(s1.granted_tools) == orig_granted1
    assert s1.pending_figure == orig_fig1

    # Case 2: Empty dataframe
    s2 = SessionState()
    s2.dataframe = pd.DataFrame(columns=["text", "category_name"])
    s2.artifacts = {"art": 2}
    s2.raw_dataset = {"raw": 2}
    s2.granted_tools = {"describe_data_tool"}
    s2.pending_figure = "old_fig"

    orig_s2_state = _snapshot_session_state(s2)
    orig_store2 = dict(s2.results_store)
    orig_art2 = dict(s2.artifacts)
    orig_raw2 = dict(s2.raw_dataset)
    orig_granted2 = set(s2.granted_tools)
    orig_fig2 = s2.pending_figure

    t2 = next(t for t in make_data_tools(s2) if t.name == "describe_data_tool")
    res2 = t2.invoke({})
    assert "error" in res2
    assert "empty" in res2["error"]
    _assert_session_state_unchanged(s2, orig_s2_state)
    assert s2.results_store == orig_store2
    assert s2.artifacts == orig_art2
    assert s2.raw_dataset == orig_raw2
    assert set(s2.granted_tools) == orig_granted2
    assert s2.pending_figure == orig_fig2

    # Case 3a: Missing 'text' column
    s3a = SessionState()
    s3a.dataframe = pd.DataFrame({"category_name": ["catA", "catB"]})
    s3a.artifacts = {"art": 3}
    s3a.raw_dataset = {"raw": 3}
    s3a.granted_tools = {"describe_data_tool"}
    s3a.pending_figure = "old_fig"

    orig_s3a_state = _snapshot_session_state(s3a)
    orig_store3a = dict(s3a.results_store)
    orig_art3a = dict(s3a.artifacts)
    orig_raw3a = dict(s3a.raw_dataset)
    orig_granted3a = set(s3a.granted_tools)
    orig_fig3a = s3a.pending_figure

    t3a = next(t for t in make_data_tools(s3a) if t.name == "describe_data_tool")
    res3a = t3a.invoke({})
    assert "error" in res3a
    assert "missing required" in res3a["error"]
    _assert_session_state_unchanged(s3a, orig_s3a_state)
    assert s3a.results_store == orig_store3a
    assert s3a.artifacts == orig_art3a
    assert s3a.raw_dataset == orig_raw3a
    assert set(s3a.granted_tools) == orig_granted3a
    assert s3a.pending_figure == orig_fig3a

    # Case 3b: Missing 'category_name' column
    s3b = SessionState()
    s3b.dataframe = pd.DataFrame({"text": ["a", "b"]})
    s3b.artifacts = {"art": 3}
    s3b.raw_dataset = {"raw": 3}
    s3b.granted_tools = {"describe_data_tool"}
    s3b.pending_figure = "old_fig"

    orig_s3b_state = _snapshot_session_state(s3b)
    orig_store3b = dict(s3b.results_store)
    orig_art3b = dict(s3b.artifacts)
    orig_raw3b = dict(s3b.raw_dataset)
    orig_granted3b = set(s3b.granted_tools)
    orig_fig3b = s3b.pending_figure

    t3b = next(t for t in make_data_tools(s3b) if t.name == "describe_data_tool")
    res3b = t3b.invoke({})
    assert "error" in res3b
    assert "missing required" in res3b["error"]
    _assert_session_state_unchanged(s3b, orig_s3b_state)
    assert s3b.results_store == orig_store3b
    assert s3b.artifacts == orig_art3b
    assert s3b.raw_dataset == orig_raw3b
    assert set(s3b.granted_tools) == orig_granted3b
    assert s3b.pending_figure == orig_fig3b

    # Case 4: Invalid text elements (None, NaN, pd.NA, numbers, lists, arrays)
    invalid_elements = [None, np.nan, pd.NA, 123, 45.6, ["a", "b"], np.array([1, 2])]
    for invalid_val in invalid_elements:
        s4 = SessionState()
        s4.dataframe = pd.DataFrame(
            {"text": ["valid text", invalid_val], "category_name": ["catA", "catA"]},
            dtype=object
        )
        s4.artifacts = {"art": 4}
        s4.raw_dataset = {"raw": 4}
        s4.granted_tools = {"describe_data_tool"}
        s4.pending_figure = "old_fig"

        orig_s4_state = _snapshot_session_state(s4)
        orig_store4 = dict(s4.results_store)
        orig_art4 = dict(s4.artifacts)
        orig_raw4 = dict(s4.raw_dataset)
        orig_granted4 = set(s4.granted_tools)
        orig_fig4 = s4.pending_figure

        t4 = next(t for t in make_data_tools(s4) if t.name == "describe_data_tool")
        res4 = t4.invoke({})
        assert "error" in res4
        assert "missing or non-string values" in res4["error"]
        _assert_session_state_unchanged(s4, orig_s4_state)
        assert s4.results_store == orig_store4
        assert s4.artifacts == orig_art4
        assert s4.raw_dataset == orig_raw4
        assert set(s4.granted_tools) == orig_granted4
        assert s4.pending_figure == orig_fig4


def test_describe_data_tool_preparation_failure_atomicity():
    # Setup base session & DTM
    session = _setup_fixture_session()
    dtm_tools = make_dtm_tools(session)
    dtm_res = next(t for t in dtm_tools if t.name == "build_dtm_tool").invoke({})
    assert "error" not in dtm_res

    old_fig = plt.figure()
    try:
        session.pending_figure = old_fig
        session.artifacts = {"art": 1}
        session.raw_dataset = {"raw": 1}
        session.granted_tools = {"describe_data_tool"}

        orig_state = _snapshot_session_state(session)
        orig_store = dict(session.results_store)
        orig_art = dict(session.artifacts)
        orig_raw = dict(session.raw_dataset)
        orig_granted = set(session.granted_tools)

        next_id_mock = Mock(side_effect=session.next_result_id)

        tools = make_data_tools(session)
        describe_tool = next(t for t in tools if t.name == "describe_data_tool")

        # Error Case 1: Series.describe raises exception
        initial_fignums1 = set(plt.get_fignums())
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(session, "next_result_id", next_id_mock)
            mp.setattr(pd.Series, "describe", lambda self: (_ for _ in ()).throw(RuntimeError("Describe failed")))
            res1 = describe_tool.invoke({})
            assert "error" in res1
            assert "Describe failed" in res1["error"]
            _assert_session_state_unchanged(session, orig_state)
            assert session.pending_figure is old_fig
            assert session.results_store == orig_store
            assert session.artifacts == orig_art
            assert session.raw_dataset == orig_raw
            assert set(session.granted_tools) == orig_granted
            assert next_id_mock.call_count == 0
            assert set(plt.get_fignums()) == initial_fignums1

        # Error Case 2: sns.boxplot raises exception
        initial_fignums2 = set(plt.get_fignums())
        next_id_mock.reset_mock()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(session, "next_result_id", next_id_mock)
            mp.setattr(sns, "boxplot", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Boxplot failed")))
            res2 = describe_tool.invoke({})
            assert "error" in res2
            assert "Boxplot failed" in res2["error"]
            _assert_session_state_unchanged(session, orig_state)
            assert session.pending_figure is old_fig
            assert session.results_store == orig_store
            assert session.artifacts == orig_art
            assert session.raw_dataset == orig_raw
            assert set(session.granted_tools) == orig_granted
            assert next_id_mock.call_count == 0
            assert set(plt.get_fignums()) == initial_fignums2

        # Error Case 3: json.dumps raises exception
        initial_fignums3 = set(plt.get_fignums())
        next_id_mock.reset_mock()
        with pytest.MonkeyPatch.context() as mp:
            orig_dumps = json.dumps
            def mock_dumps(obj, *args, **kwargs):
                if isinstance(obj, dict) and "overall" in obj and "per_category" in obj:
                    raise ValueError("JSON serialization failed")
                return orig_dumps(obj, *args, **kwargs)

            mp.setattr(session, "next_result_id", next_id_mock)
            mp.setattr(json, "dumps", mock_dumps)
            res3 = describe_tool.invoke({})
            assert "error" in res3
            assert "JSON serialization failed" in res3["error"]
            _assert_session_state_unchanged(session, orig_state)
            assert session.pending_figure is old_fig
            assert session.results_store == orig_store
            assert session.artifacts == orig_art
            assert session.raw_dataset == orig_raw
            assert set(session.granted_tools) == orig_granted
            assert next_id_mock.call_count == 0
            assert set(plt.get_fignums()) == initial_fignums3

    finally:
        plt.close(old_fig)
