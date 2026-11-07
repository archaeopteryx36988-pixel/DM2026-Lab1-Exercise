import json
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def _setup_fixture_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session

def test_cosine_similarity_known_answers_and_records():
    session = _setup_fixture_session()

    # Save original state to check non-mutation later
    orig_df = session.dataframe.copy()
    orig_fm = session.feature_matrix.copy()
    orig_labels = session.labels.copy() if session.labels is not None else None
    orig_vocab = list(session.artifacts["count_vectorizer"].vocabulary_.items())

    tools = make_exploration_tools(session)
    cosine_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    # 1. Known Answer 16 Checks
    # 0 vs 1 -> 0.9847
    res_01 = cosine_tool.invoke({"doc_index1": 0, "doc_index2": 1})
    assert "error" not in res_01
    assert res_01["doc_index1"] == 0
    assert res_01["doc_index2"] == 1
    assert res_01["cosine_similarity"] == pytest.approx(0.9847, abs=1e-4)

    # 0 vs 3 -> 0.0909
    res_03 = cosine_tool.invoke({"doc_index1": 0, "doc_index2": 3})
    assert "error" not in res_03
    assert res_03["cosine_similarity"] == pytest.approx(0.0909, abs=1e-4)

    # 3 vs 4 -> 0.9847
    res_34 = cosine_tool.invoke({"doc_index1": 3, "doc_index2": 4})
    assert "error" not in res_34
    assert res_34["cosine_similarity"] == pytest.approx(0.9847, abs=1e-4)

    # 0 vs 7 -> exactly 0.0 (empty text doc)
    res_07 = cosine_tool.invoke({"doc_index1": 0, "doc_index2": 7})
    assert "error" not in res_07
    assert res_07["cosine_similarity"] == 0.0

    # 2. Check result store & record structure
    rid = res_01["result_id"]
    stored = session.results_store[rid]
    record = stored["record"]

    assert record.args == {"doc_index1": 0, "doc_index2": 1}
    assert record.summary == res_01
    json.dumps(res_01)

    # 4. State non-mutation check
    assert_frame_equal(session.dataframe, orig_df)
    assert session.feature_matrix.shape == orig_fm.shape
    np.testing.assert_array_equal(session.feature_matrix.toarray(), orig_fm.toarray())
    if orig_labels is not None:
        np.testing.assert_array_equal(session.labels, orig_labels)
    assert list(session.artifacts["count_vectorizer"].vocabulary_.items()) == orig_vocab

def test_cosine_similarity_independent_of_feature_matrix():
    # 5. Modify feature_matrix to 1 column, verify cosine_similarity still uses original vectorizer
    session = _setup_fixture_session()
    session.feature_matrix = np.zeros((len(session.dataframe), 1))

    tools = make_exploration_tools(session)
    cosine_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    res_01 = cosine_tool.invoke({"doc_index1": 0, "doc_index2": 1})
    assert "error" not in res_01
    assert res_01["cosine_similarity"] == pytest.approx(0.9847, abs=1e-4)

def test_cosine_similarity_errors():
    # 3. Missing dataframe
    s1 = SessionState()
    tools1 = make_exploration_tools(s1)
    t1 = next(t for t in tools1 if t.name == "cosine_similarity_tool")
    r1 = t1.invoke({"doc_index1": 0, "doc_index2": 1})
    assert "error" in r1

    # Missing 'text' column
    s2 = SessionState()
    s2.dataframe = pd.DataFrame({"other_col": [1, 2]})
    s2.artifacts["count_vectorizer"] = "dummy"
    tools2 = make_exploration_tools(s2)
    t2 = next(t for t in tools2 if t.name == "cosine_similarity_tool")
    r2 = t2.invoke({"doc_index1": 0, "doc_index2": 1})
    assert "error" in r2

    # Missing count_vectorizer
    s3 = SessionState()
    s3.dataframe = pd.DataFrame({"text": ["a", "b"]})
    tools3 = make_exploration_tools(s3)
    t3 = next(t for t in tools3 if t.name == "cosine_similarity_tool")
    r3 = t3.invoke({"doc_index1": 0, "doc_index2": 1})
    assert "error" in r3

    # Out of bounds / Negative index for doc_index1 and doc_index2
    session_valid = _setup_fixture_session()
    tools_v = make_exploration_tools(session_valid)
    t_v = next(t for t in tools_v if t.name == "cosine_similarity_tool")

    assert "error" in t_v.invoke({"doc_index1": -1, "doc_index2": 1})
    assert "error" in t_v.invoke({"doc_index1": 100, "doc_index2": 1})
    assert "error" in t_v.invoke({"doc_index1": 0, "doc_index2": -1})
    assert "error" in t_v.invoke({"doc_index1": 0, "doc_index2": 100})

def test_feature_correlation_matrix_known_answer_and_records():
    session = _setup_fixture_session()
    session.pending_figure = None

    orig_df = session.dataframe.copy()
    orig_fm = session.feature_matrix.copy()
    orig_fn = list(session.feature_names)
    orig_labels = session.labels.copy() if session.labels is not None else None

    tools = make_exploration_tools(session)
    corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")

    res = corr_tool.invoke({})
    assert "error" not in res

    # Check pending_figure is created
    assert session.pending_figure is not None

    # Check Known Answer 19
    expected_names = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["feature_names"] == expected_names

    corr_matrix = np.array(res["correlation_matrix"], dtype=float)
    expected_matrix = np.array([
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000]
    ])
    np.testing.assert_allclose(corr_matrix, expected_matrix, atol=1e-3)

    # Check record storage & json serialization without NaN
    rid = res["result_id"]
    stored = session.results_store[rid]
    record = stored["record"]

    assert record.args == {}
    assert record.summary == res
    json.dumps(res, allow_nan=False)

    # Check non-mutation
    assert_frame_equal(session.dataframe, orig_df)
    assert session.feature_matrix.shape == orig_fm.shape
    np.testing.assert_array_equal(session.feature_matrix.toarray(), orig_fm.toarray())
    assert session.feature_names == orig_fn
    if orig_labels is not None:
        np.testing.assert_array_equal(session.labels, orig_labels)

def test_feature_correlation_matrix_top20_cutoff():
    session = SessionState()
    # Create 25 features with distinct variances
    n_docs = 10
    n_feats = 25
    matrix = np.zeros((n_docs, n_feats))
    for i in range(n_feats):
        # Column i has variance proportional to (i + 1)^2
        matrix[:, i] = np.linspace(0, (i + 1), n_docs)

    feature_names = [f"f_{i}" for i in range(n_feats)]
    session.feature_matrix = matrix
    session.feature_names = feature_names

    tools = make_exploration_tools(session)
    corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")

    res = corr_tool.invoke({})
    assert "error" not in res

    # Selected top 20 features should be f_24 down to f_5
    expected_top20_names = [f"f_{i}" for i in range(24, 4, -1)]
    assert len(res["feature_names"]) == 20
    assert res["feature_names"] == expected_top20_names

    corr_m = np.array(res["correlation_matrix"], dtype=float)
    assert corr_m.shape == (20, 20)

def test_feature_correlation_matrix_edge_cases():
    # Single non-constant feature -> [[1.0]]
    s1 = SessionState()
    s1.feature_matrix = np.array([[1.0], [2.0], [3.0]])
    s1.feature_names = ["feat1"]
    t1 = next(t for t in make_exploration_tools(s1) if t.name == "feature_correlation_matrix_tool")
    r1 = t1.invoke({})
    assert "error" not in r1
    assert r1["correlation_matrix"] == [[1.0]]

    # Constant feature -> correlation matrix entry is None
    s2 = SessionState()
    s2.feature_matrix = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    s2.feature_names = ["feat1", "const_feat"]
    t2 = next(t for t in make_exploration_tools(s2) if t.name == "feature_correlation_matrix_tool")
    r2 = t2.invoke({})
    assert "error" not in r2
    assert r2["correlation_matrix"][0][1] is None
    assert r2["correlation_matrix"][1][0] is None
    assert r2["correlation_matrix"][1][1] is None
    json.dumps(r2, allow_nan=False)

    # Missing feature matrix
    s3 = SessionState()
    t3 = next(t for t in make_exploration_tools(s3) if t.name == "feature_correlation_matrix_tool")
    assert "error" in t3.invoke({})

    # Empty feature matrix (0 cols)
    s4 = SessionState()
    s4.feature_matrix = np.zeros((3, 0))
    s4.feature_names = []
    t4 = next(t for t in make_exploration_tools(s4) if t.name == "feature_correlation_matrix_tool")
    assert "error" in t4.invoke({})

    # Single document (less than 2 docs)
    s5 = SessionState()
    s5.feature_matrix = np.array([[1.0, 2.0]])
    s5.feature_names = ["f1", "f2"]
    t5 = next(t for t in make_exploration_tools(s5) if t.name == "feature_correlation_matrix_tool")
    assert "error" in t5.invoke({})
