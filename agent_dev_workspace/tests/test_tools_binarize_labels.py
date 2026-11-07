import json
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from agent_pipeline.session_state import SessionState
from tools_reduction import make_tools

def test_binarize_labels_tool_basic():
    session = SessionState()
    session.labels = np.array(["B", "A", "B"])
    
    # Preserve original labels and create dummy non-empty feature_matrix
    orig_labels = session.labels.copy()
    session.feature_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    orig_feature_matrix = session.feature_matrix.copy()

    tools = make_tools(session)
    tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = tool.invoke({})

    assert "error" not in res
    assert res["n_samples"] == 3
    assert res["categories"] == ["A", "B"]
    assert res["one_hot_labels"] == [[0, 1], [1, 0], [0, 1]]

    result_id = res["result_id"]
    stored = session.results_store[result_id]
    record = stored["record"]

    assert record.args == {}
    assert record.summary == res

    # Ensure summary is JSON serializable
    json.dumps(res)

    full_report = stored["full_report"]
    expected_df = pd.DataFrame([[0, 1], [1, 0], [0, 1]], columns=["A", "B"])
    assert_frame_equal(full_report, expected_df)

    # Check session artifacts
    artifacts = session.artifacts["binarized_labels"]
    assert artifacts["categories"] == ["A", "B"]
    np.testing.assert_array_equal(artifacts["one_hot_matrix"], np.array([[0, 1], [1, 0], [0, 1]]))

    # Ensure session labels and feature_matrix remain unchanged
    np.testing.assert_array_equal(session.labels, orig_labels)
    np.testing.assert_array_equal(session.feature_matrix, orig_feature_matrix)

def test_binarize_labels_tool_fallback_to_dataframe():
    session = SessionState()
    session.labels = None
    session.dataframe = pd.DataFrame({"category_name": ["catX", "catY"]})
    orig_df = session.dataframe.copy()

    tools = make_tools(session)
    tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = tool.invoke({})

    assert "error" not in res
    assert res["categories"] == ["catX", "catY"]
    assert res["one_hot_labels"] == [[1, 0], [0, 1]]

    # Ensure dataframe remains unchanged
    assert_frame_equal(session.dataframe, orig_df)

def test_binarize_labels_tool_single_category():
    session = SessionState()
    session.labels = np.array(["catA", "catA"])

    tools = make_tools(session)
    tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = tool.invoke({})

    assert "error" not in res
    assert res["n_samples"] == 2
    assert res["categories"] == ["catA"]
    assert res["one_hot_labels"] == [[1], [1]]

def test_binarize_labels_tool_errors():
    # Case 1: Missing labels entirely
    session1 = SessionState()
    session1.labels = None
    session1.dataframe = None

    tools1 = make_tools(session1)
    tool1 = next(t for t in tools1 if t.name == "binarize_labels_tool")

    res1 = tool1.invoke({})
    assert "error" in res1

    # Case 2: Empty labels array
    session2 = SessionState()
    session2.labels = np.array([])

    tools2 = make_tools(session2)
    tool2 = next(t for t in tools2 if t.name == "binarize_labels_tool")

    res2 = tool2.invoke({})
    assert "error" in res2
