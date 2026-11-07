import numpy as np
import pytest
import scipy.sparse as sp
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """Test build_dtm_tool against TEST_FIXTURE.md Known Answer #4.

    Asserts that building DTM on agent_dev/sample_fixture.csv yields:
    - Vocabulary: ['alpha', 'always', 'beta', 'delta', 'gamma'] (5 terms)
    - Matrix shape: 8 documents x 5 vocabulary
    - Non-zero count: 21
    - Total elements: 40
    - Sparsity percentage: 47.5%
    - session.labels populated with category names ('catA', 'catB')
    """
    # 1. Initialize single session instance
    session = SessionState()

    # 2. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # 3. Load sample fixture dataset
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 4. Construct tools from tools_dtm.py
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 5. Invoke build_dtm_tool with defaults
    result = build_dtm_tool.invoke({})

    # 6. Assert summary outputs matching Known Answer #4
    assert result["n_documents"] == 8
    assert result["n_vocabulary"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # 7. Assert session attributes
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.artifacts.get("count_vectorizer") is not None

    # 8. Verify session.labels contains string category names ('catA', 'catB')
    assert len(session.labels) == 8
    assert set(session.labels) == {"catA", "catB"}


def test_term_frequency_tool_known_answer_14():
    """Test term_frequency_tool against TEST_FIXTURE.md Known Answer #14.

    Asserts that aggregating term frequencies yields:
    - Total terms: 5, Total frequency: 27
    - Correct term counts: alpha=6, always=7, beta=3, delta=4, gamma=7
    - top_n=2 truncates summary frequencies to top 2 ('always': 7, 'gamma': 7) while
      full_report retains all 5 terms
    - top_n out of bounds (0 or 101) raises ValueError
    """
    # 1. Fresh session for this test
    session = SessionState()

    # 2. Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 3. Build DTM first
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(
        t for t in dtm_tools if t.name == "term_frequency_tool"
    )
    build_dtm_tool.invoke({})

    # 4. Invoke term_frequency_tool with defaults (top_n=20)
    result = term_frequency_tool.invoke({})

    assert result["total_terms"] == 5
    assert result["total_frequency"] == 27
    assert result["frequencies"] == {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3,
    }

    # 5. Invoke with top_n=2
    result_top2 = term_frequency_tool.invoke({"top_n": 2})
    assert result_top2["total_terms"] == 5
    assert result_top2["total_frequency"] == 27
    assert len(result_top2["frequencies"]) == 2
    assert result_top2["frequencies"] == {"always": 7, "gamma": 7}

    # Verify full_report in results_store contains all 5 terms
    stored_result = session.results_store[result_top2["result_id"]]
    full_report = stored_result["full_report"]
    assert full_report is not None
    assert len(full_report) == 5

    # 6. Verify error handling for out-of-bounds top_n
    with pytest.raises(ValueError, match="top_n must be an integer between 1 and 100"):
        term_frequency_tool.invoke({"top_n": 0})

    with pytest.raises(ValueError, match="top_n must be an integer between 1 and 100"):
        term_frequency_tool.invoke({"top_n": 101})


def test_dtm_heatmap_tool_known_answer_21():
    """Test dtm_heatmap_tool against TEST_FIXTURE.md Known Answer #21.

    Asserts that positional DTM heatmap slicing yields:
    - Default (20x20): Full 8x5 matrix matching Known Answer #21
    - Sliced (2x3): Top-left 3x2 matrix [[3, 1], [2, 1], [1, 1]] with terms ['alpha', 'always']
    - session.pending_figure is assigned and original DTM is unaltered (shape and values preserved)
    - Non-positive parameters raise ValueError
    - Large DTMs (>50x50) are correctly clamped to 50x50 maximum display limit
    """
    # 1. Fresh session
    session = SessionState()

    # 2. Load dataset and build DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(
        t for t in dtm_tools if t.name == "dtm_heatmap_tool"
    )
    build_dtm_tool.invoke({})

    # Save a deep copy of original DTM matrix for immutability assertions
    original_dtm = session.feature_matrix.copy()

    # 3. Default call (20x20 requested -> returns full 8x5)
    result_default = dtm_heatmap_tool.invoke({})
    assert result_default["n_documents_shown"] == 8
    assert result_default["n_terms_shown"] == 5
    assert result_default["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert result_default["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # Verify original DTM shape and exact values were not mutated
    assert session.feature_matrix.shape == original_dtm.shape
    assert (session.feature_matrix != original_dtm).nnz == 0

    # 4. Explicit slice call (n_terms=2, n_documents=3)
    result_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert result_sliced["n_documents_shown"] == 3
    assert result_sliced["n_terms_shown"] == 2
    assert result_sliced["terms"] == ["alpha", "always"]
    assert result_sliced["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]

    # Verify original DTM shape and exact values remain unchanged after slicing
    assert session.feature_matrix.shape == original_dtm.shape
    assert (session.feature_matrix != original_dtm).nnz == 0

    # 5. Invalid parameters check (0 or negative)
    with pytest.raises(
        ValueError, match="n_terms and n_documents must be positive integers."
    ):
        dtm_heatmap_tool.invoke({"n_terms": 0, "n_documents": 20})

    with pytest.raises(
        ValueError, match="n_terms and n_documents must be positive integers."
    ):
        dtm_heatmap_tool.invoke({"n_terms": 10, "n_documents": -5})

    # 6. Test 50x50 clamping limit with a mock large integer DTM (60x60)
    session_large = SessionState()
    session_large.feature_matrix = sp.csr_matrix((60, 60), dtype=int)
    session_large.feature_names = [f"term_{i}" for i in range(60)]
    tools_large = make_tools(session_large)
    dtm_heatmap_large = next(
        t for t in tools_large if t.name == "dtm_heatmap_tool"
    )

    result_clamped = dtm_heatmap_large.invoke({"n_terms": 60, "n_documents": 60})
    assert result_clamped["n_documents_shown"] == 50
    assert result_clamped["n_terms_shown"] == 50
    assert len(result_clamped["matrix_slice"]) == 50
    assert len(result_clamped["matrix_slice"][0]) == 50
