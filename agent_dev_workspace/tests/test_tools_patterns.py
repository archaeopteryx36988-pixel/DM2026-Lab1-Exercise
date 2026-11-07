import json
import pandas as pd
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tool_known_answers_17_and_20():
    """Test mine_patterns_tool against TEST_FIXTURE.md Known Answers #17 and #20.

    Asserts that:
    1. fpgrowth + tfidf (Known Answer #17):
       - catA -> {alpha} with support 3 (total_patterns=1)
       - catB -> {gamma} with support 4 (total_patterns=1)
    2. topk + tfidf (Known Answer #20):
       - k=1, 2, 3 on catA -> {alpha} support 3 (total_patterns=1)
       - k=1, 2, 3 on catB -> {gamma} support 4 (total_patterns=1)
    3. maxfpgrowth + tfidf (Known Answer #20):
       - Returns 0 patterns (total_patterns=0, note="no patterns found") for both catA and catB at min_sup=1 and 4,
         which is real PAMI MaxFPGrowth behavior on a single-surviving-term database.
    4. variance and term_frequency filtering on this fixture keep 0 terms and return clean empty results
       with note="no terms survived filtering".
    5. All-blank text or all-stopword text handled safely without raising exceptions.
    6. State overwriting: High min_sup clears old artifact and sets total_patterns=0. Invalid min_sup=0/k=0 raise ValueError.
    7. Payload capping: 63 combinatorial patterns capped at 50 in summary while artifact holds full 63 patterns.
    8. session.dataframe and global DTM remain immutable, summary is JSON serializable.
    """
    # 1. Initialize session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build global DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record initial state for immutability check
    original_df = session.dataframe.copy()
    original_matrix = session.feature_matrix.copy()
    original_feature_names = list(session.feature_names)
    original_labels = list(session.labels)

    # Construct patterns tools
    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # --- 2. FPGrowth + TF-IDF (Known Answer #17) ---
    res_fpg_catA = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1}
    )
    assert res_fpg_catA["total_patterns"] == 1
    assert res_fpg_catA["patterns"] == [{"pattern": ["alpha"], "support": 3}]
    assert session.artifacts.get("patterns_catA_tfidf_fpgrowth") == [{"pattern": ["alpha"], "support": 3}]

    res_fpg_catB = mine_patterns_tool.invoke(
        {"category_name": "catB", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1}
    )
    assert res_fpg_catB["total_patterns"] == 1
    assert res_fpg_catB["patterns"] == [{"pattern": ["gamma"], "support": 4}]
    assert session.artifacts.get("patterns_catB_tfidf_fpgrowth") == [{"pattern": ["gamma"], "support": 4}]

    # --- 3. Top-K + TF-IDF (Known Answer #20) ---
    for k_val in [1, 2, 3]:
        res_topk_catA = mine_patterns_tool.invoke(
            {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": k_val}
        )
        assert res_topk_catA["total_patterns"] == 1
        assert res_topk_catA["patterns"] == [{"pattern": ["alpha"], "support": 3}]

        res_topk_catB = mine_patterns_tool.invoke(
            {"category_name": "catB", "filtering_method": "tfidf", "algorithm": "topk", "k": k_val}
        )
        assert res_topk_catB["total_patterns"] == 1
        assert res_topk_catB["patterns"] == [{"pattern": ["gamma"], "support": 4}]

    # --- 4. MaxFPGrowth + TF-IDF (Known Answer #20) ---
    # Exactly 0 patterns returned by PAMI MaxFPGrowth on single surviving term
    for min_sup_val in [1, 4]:
        res_max_catA = mine_patterns_tool.invoke(
            {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": min_sup_val}
        )
        assert res_max_catA["total_patterns"] == 0
        assert res_max_catA["patterns"] == []
        assert res_max_catA["note"] == "no patterns found"

        res_max_catB = mine_patterns_tool.invoke(
            {"category_name": "catB", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": min_sup_val}
        )
        assert res_max_catB["total_patterns"] == 0
        assert res_max_catB["patterns"] == []
        assert res_max_catB["note"] == "no patterns found"

    # --- 5. Variance and Term Frequency percentile filtering (0 terms survive) ---
    res_var_catA = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "variance", "algorithm": "fpgrowth", "min_sup": 1}
    )
    assert res_var_catA["total_patterns"] == 0
    assert res_var_catA["patterns"] == []
    assert res_var_catA["note"] == "no terms survived filtering"

    res_tf_catB = mine_patterns_tool.invoke(
        {"category_name": "catB", "filtering_method": "term_frequency", "algorithm": "fpgrowth", "min_sup": 1}
    )
    assert res_tf_catB["total_patterns"] == 0
    assert res_tf_catB["patterns"] == []
    assert res_tf_catB["note"] == "no terms survived filtering"

    # --- 6. Edge cases: All-blank / All-stopwords text ---
    session_empty = SessionState()
    session_empty.dataframe = pd.DataFrame(
        {"text": ["", "   ", "the is an a"], "category_name": ["catX", "catX", "catX"]}
    )
    tools_empty = make_patterns_tools(session_empty)
    mine_empty = next(t for t in tools_empty if t.name == "mine_patterns_tool")

    res_empty = mine_empty.invoke(
        {"category_name": "catX", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1}
    )
    assert res_empty["total_patterns"] == 0
    assert res_empty["note"] == "no terms survived stopword removal"

    # --- 7. Artifact clearing & Invalid parameter checks ---
    # High min_sup=10 on catA clears old artifact
    res_high_sup = mine_patterns_tool.invoke(
        {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 10}
    )
    assert res_high_sup["total_patterns"] == 0
    assert res_high_sup["patterns"] == []
    assert res_high_sup["note"] == "no patterns found"
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == []

    # Invalid parameter error handling
    with pytest.raises(ValueError, match="min_sup must be an integer >= 1."):
        mine_patterns_tool.invoke(
            {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 0}
        )

    with pytest.raises(ValueError, match="k must be an integer >= 1."):
        mine_patterns_tool.invoke(
            {"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": 0}
        )

    # --- 8. Combinatorial payload capping test (63 patterns -> 50 in summary) ---
    session_combo = SessionState()
    session_combo.dataframe = pd.DataFrame(
        {
            "text": ["m1 m2 m3 m4 m5 m6", "m1 m2 m3 m4 m5 m6"],
            "category_name": ["catCombo", "catCombo"],
        }
    )
    tools_combo = make_patterns_tools(session_combo)
    mine_combo = next(t for t in tools_combo if t.name == "mine_patterns_tool")

    res_combo = mine_combo.invoke(
        {"category_name": "catCombo", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1}
    )
    # 2^6 - 1 = 63 combinations
    assert res_combo["total_patterns"] == 63
    assert len(res_combo["patterns"]) == 50  # Capped at 50 in summary
    assert len(session_combo.artifacts["patterns_catCombo_tfidf_fpgrowth"]) == 63  # Full in artifact

    # --- 9. Immutability and JSON serializability checks ---
    assert session.dataframe.equals(original_df)
    assert session.feature_matrix.shape == original_matrix.shape
    assert (session.feature_matrix != original_matrix).nnz == 0
    assert session.feature_names == original_feature_names
    assert list(session.labels) == original_labels

    # Verify JSON serializability of summary and stored result
    stored_result = session.results_store[res_fpg_catA["result_id"]]
    record = stored_result["record"]
    assert json.dumps(record.summary) is not None
    assert record.args["category_name"] == "catA"
    assert record.args["filtering_method"] == "tfidf"
    assert record.args["algorithm"] == "fpgrowth"
    assert record.args["min_sup"] == 1
    assert record.args["k"] == 10
