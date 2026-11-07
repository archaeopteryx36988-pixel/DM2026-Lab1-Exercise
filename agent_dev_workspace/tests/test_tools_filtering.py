import numpy as np
import pytest
import scipy.sparse as sp
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    """Test variance_filter_tool against TEST_FIXTURE.md Known Answer #5.

    Asserts that:
    - Feature variances match Known Answer #5 (alpha: 1.1875, gamma: 1.109375, delta: 0.25, beta: 0.234375, always: 0.109375)
    - threshold=0.15 keeps exactly 4 features (alpha, beta, delta, gamma) and removes 'always' (1 feature)
    - full_report is a DataFrame sorted descending by variance containing all 5 features
    - session.feature_matrix, session.feature_names, and their values remain immutable
    - Constant columns across all documents yield exactly 0.0 variance and are excluded at threshold=0.0
    """
    # 1. Single session instance for loading data and building DTM
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

    # 3. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record initial state and deep copy DTM for immutability check
    original_matrix = session.feature_matrix.copy()
    original_feature_names = list(session.feature_names)

    # 4. Construct filtering tools and invoke variance_filter_tool with threshold=0.15
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(
        t for t in filtering_tools if t.name == "variance_filter_tool"
    )

    result = variance_filter_tool.invoke({"threshold": 0.15})

    # 5. Assert summary matching Known Answer #5
    assert result["total_features"] == 5
    assert result["kept_features_count"] == 4
    assert result["removed_features_count"] == 1
    assert result["threshold"] == 0.15

    # Verify kept features in session.artifacts['variance_filter']
    kept_features = session.artifacts.get("variance_filter")
    assert kept_features is not None
    assert set(kept_features) == {"alpha", "beta", "delta", "gamma"}
    assert "always" not in kept_features

    # 6. Verify full_report DataFrame contents and descending order
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    assert full_report is not None
    assert len(full_report) == 5

    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375,
    }

    # Assert exact variance values within tolerance
    for term, expected_var in expected_variances.items():
        actual_var = full_report.loc[
            full_report["feature"] == term, "variance"
        ].values[0]
        assert actual_var == pytest.approx(expected_var, abs=1e-4)

    # Assert descending sort order of full_report
    report_variances = full_report["variance"].tolist()
    assert report_variances == sorted(report_variances, reverse=True)

    # 7. Assert DTM shape, feature_names, and values remain unaltered
    assert session.feature_matrix.shape == original_matrix.shape
    assert session.feature_names == original_feature_names
    assert (session.feature_matrix != original_matrix).nnz == 0

    # 8. Test constant column case: 7 documents with term count = 1 in every document
    session_constant = SessionState()
    # 7x1 sparse matrix with all entries equal to 1
    session_constant.feature_matrix = sp.csr_matrix(
        [[1], [1], [1], [1], [1], [1], [1]], dtype=int
    )
    session_constant.feature_names = ["constant_term"]

    tools_constant = make_filtering_tools(session_constant)
    variance_filter_constant = next(
        t for t in tools_constant if t.name == "variance_filter_tool"
    )

    result_const = variance_filter_constant.invoke({"threshold": 0.0})
    assert result_const["total_features"] == 1
    assert result_const["kept_features_count"] == 0
    assert result_const["removed_features_count"] == 1

    stored_const = session_constant.results_store[result_const["result_id"]]
    const_report = stored_const["full_report"]
    assert const_report.iloc[0]["variance"] == pytest.approx(0.0, abs=1e-9)
    assert session_constant.artifacts["variance_filter"] == []


def test_pearson_filter_tool_known_answer_6():
    """Test pearson_filter_tool against TEST_FIXTURE.md Known Answer #6.

    Asserts that:
    - Pearson correlations for target_class='catB' match Known Answer #6:
      delta: 1.0000, gamma: 0.8307, always: 0.3780, alpha: -0.6882, beta: -0.7746
    - threshold=0.5 keeps exactly 2 features (delta, gamma) and removes 3 features
    - full_report contains all 5 features sorted descending by pearson_r
    - session.feature_matrix, feature_names, and labels remain immutable
    - Constant feature [1, 1, 1] with labels ["catB", "catA", "catA"] returns r=0.0 (finite, 0 kept at threshold=0.0)
    - All-identical labels ["catA", "catA", "catA"] with feature [1, 0, 0] returns r=0.0 (finite, 0 kept at threshold=0.0)
    - Perfectly correlated feature [1, 0, 0] against ["catB", "catA", "catA"] yields r=1.0,
      and threshold=1.0 keeps 0 features due to strict r > threshold filtering
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record initial state for immutability assertions
    original_matrix = session.feature_matrix.copy()
    original_feature_names = list(session.feature_names)
    original_labels = list(session.labels)

    # 3. Construct filtering tools and invoke pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(
        t for t in filtering_tools if t.name == "pearson_filter_tool"
    )

    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    # 4. Assert summary and kept features
    assert result["total_features"] == 5
    assert result["kept_features_count"] == 2
    assert result["removed_features_count"] == 3
    assert result["target_class"] == "catB"
    assert result["threshold"] == 0.5

    kept_features = session.artifacts.get("pearson_filter")
    assert kept_features is not None
    assert set(kept_features) == {"delta", "gamma"}

    # 5. Verify full_report contents and descending sort order
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    assert full_report is not None
    assert len(full_report) == 5

    expected_correlations = {
        "delta": 1.0000,
        "gamma": 0.8307,
        "always": 0.3780,
        "alpha": -0.6882,
        "beta": -0.7746,
    }

    for term, expected_r in expected_correlations.items():
        actual_r = full_report.loc[
            full_report["feature"] == term, "pearson_r"
        ].values[0]
        assert actual_r == pytest.approx(expected_r, abs=1e-4)

    report_r_values = full_report["pearson_r"].tolist()
    assert report_r_values == sorted(report_r_values, reverse=True)

    # 6. Assert immutability of session attributes
    assert session.feature_matrix.shape == original_matrix.shape
    assert (session.feature_matrix != original_matrix).nnz == 0
    assert session.feature_names == original_feature_names
    assert list(session.labels) == original_labels

    # 7a. Constant feature test: feature_matrix = [[1], [1], [1]], labels = ["catB", "catA", "catA"]
    session_const_feat = SessionState()
    session_const_feat.feature_matrix = sp.csr_matrix([[1], [1], [1]], dtype=int)
    session_const_feat.feature_names = ["const_feature"]
    session_const_feat.set_labels(np.array(["catB", "catA", "catA"]))

    tools_const_feat = make_filtering_tools(session_const_feat)
    pearson_const_feat = next(
        t for t in tools_const_feat if t.name == "pearson_filter_tool"
    )

    res_const_feat = pearson_const_feat.invoke({"target_class": "catB", "threshold": 0.0})
    stored_cf = session_const_feat.results_store[res_const_feat["result_id"]]
    r_const_feat = stored_cf["full_report"].iloc[0]["pearson_r"]

    assert np.isfinite(r_const_feat)
    assert r_const_feat == pytest.approx(0.0, abs=1e-9)
    assert res_const_feat["kept_features_count"] == 0

    # 7b. All-identical labels test: feature_matrix = [[1], [0], [0]], labels = ["catA", "catA", "catA"]
    session_ident_labels = SessionState()
    session_ident_labels.feature_matrix = sp.csr_matrix([[1], [0], [0]], dtype=int)
    session_ident_labels.feature_names = ["var_feature"]
    session_ident_labels.set_labels(np.array(["catA", "catA", "catA"]))

    tools_ident_labels = make_filtering_tools(session_ident_labels)
    pearson_ident_labels = next(
        t for t in tools_ident_labels if t.name == "pearson_filter_tool"
    )

    res_ident_labels = pearson_ident_labels.invoke({"target_class": "catA", "threshold": 0.0})
    stored_il = session_ident_labels.results_store[res_ident_labels["result_id"]]
    r_ident_labels = stored_il["full_report"].iloc[0]["pearson_r"]

    assert np.isfinite(r_ident_labels)
    assert r_ident_labels == pytest.approx(0.0, abs=1e-9)
    assert res_ident_labels["kept_features_count"] == 0

    # 8. Boundary test: Perfectly correlated feature [1, 0, 0] with ["catB", "catA", "catA"]
    session_perf = SessionState()
    session_perf.feature_matrix = sp.csr_matrix([[1], [0], [0]], dtype=int)
    session_perf.feature_names = ["perf_feature"]
    session_perf.set_labels(np.array(["catB", "catA", "catA"]))

    tools_perf = make_filtering_tools(session_perf)
    pearson_perf = next(
        t for t in tools_perf if t.name == "pearson_filter_tool"
    )

    # Check r is bounded in [-1.0, 1.0] and exactly 1.0 for perfect correlation
    result_perf_default = pearson_perf.invoke({"target_class": "catB", "threshold": 0.0})
    stored_perf = session_perf.results_store[result_perf_default["result_id"]]
    perf_report = stored_perf["full_report"]
    actual_perf_r = perf_report.iloc[0]["pearson_r"]

    assert -1.0 <= actual_perf_r <= 1.0
    assert actual_perf_r == pytest.approx(1.0, abs=1e-9)

    # With threshold=1.0, r > threshold (1.0 > 1.0) is False -> keeps 0 features
    result_perf_thresh1 = pearson_perf.invoke({"target_class": "catB", "threshold": 1.0})
    assert result_perf_thresh1["kept_features_count"] == 0
    assert session_perf.artifacts["pearson_filter"] == []


def test_spearman_filter_tool_known_answer_6():
    """Test spearman_filter_tool against TEST_FIXTURE.md Known Answer #6.

    Asserts that:
    - Spearman correlations for target_class='catB' match Known Answer #6:
      delta: 1.0000, gamma: 0.936329, always: 0.377964, alpha: -0.7500, beta: -0.774597
    - threshold=0.5 keeps exactly 2 features (delta, gamma) and removes 3 features
    - full_report contains all 5 features sorted descending by spearman_r with columns ['feature', 'spearman_r']
    - session.feature_matrix, feature_names, and labels remain immutable
    - Constant feature [1, 1, 1] with labels ["catB", "catA", "catA"] returns r=0.0 (finite, 0 kept at threshold=0.0)
    - All-identical labels ["catA", "catA", "catA"] with feature [1, 0, 0] returns r=0.0 (finite, 0 kept at threshold=0.0)
    - Perfectly correlated feature [1, 0, 0] against ["catB", "catA", "catA"] yields r=1.0,
      and threshold=1.0 keeps 0 features due to strict r > threshold filtering
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Record initial state for immutability assertions
    original_matrix = session.feature_matrix.copy()
    original_feature_names = list(session.feature_names)
    original_labels = list(session.labels)

    # 3. Construct filtering tools and invoke spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(
        t for t in filtering_tools if t.name == "spearman_filter_tool"
    )

    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    # 4. Assert summary and kept features
    assert result["total_features"] == 5
    assert result["kept_features_count"] == 2
    assert result["removed_features_count"] == 3
    assert result["target_class"] == "catB"
    assert result["threshold"] == 0.5

    kept_features = session.artifacts.get("spearman_filter")
    assert kept_features is not None
    assert set(kept_features) == {"delta", "gamma"}

    # 5. Verify full_report contents, columns, and descending sort order
    stored_result = session.results_store[result["result_id"]]
    full_report = stored_result["full_report"]
    assert full_report is not None
    assert len(full_report) == 5
    assert list(full_report.columns) == ["feature", "spearman_r"]

    expected_correlations = {
        "delta": 1.0000,
        "gamma": 0.936329,
        "always": 0.377964,
        "alpha": -0.7500,
        "beta": -0.774597,
    }

    for term, expected_r in expected_correlations.items():
        actual_r = full_report.loc[
            full_report["feature"] == term, "spearman_r"
        ].values[0]
        assert actual_r == pytest.approx(expected_r, abs=1e-4)

    report_r_values = full_report["spearman_r"].tolist()
    assert report_r_values == sorted(report_r_values, reverse=True)

    # 6. Assert immutability of session attributes
    assert session.feature_matrix.shape == original_matrix.shape
    assert (session.feature_matrix != original_matrix).nnz == 0
    assert session.feature_names == original_feature_names
    assert list(session.labels) == original_labels

    # 7a. Constant feature test: feature_matrix = [[1], [1], [1]], labels = ["catB", "catA", "catA"]
    session_const_feat = SessionState()
    session_const_feat.feature_matrix = sp.csr_matrix([[1], [1], [1]], dtype=int)
    session_const_feat.feature_names = ["const_feature"]
    session_const_feat.set_labels(np.array(["catB", "catA", "catA"]))

    tools_const_feat = make_filtering_tools(session_const_feat)
    spearman_const_feat = next(
        t for t in tools_const_feat if t.name == "spearman_filter_tool"
    )

    res_const_feat = spearman_const_feat.invoke({"target_class": "catB", "threshold": 0.0})
    stored_cf = session_const_feat.results_store[res_const_feat["result_id"]]
    r_const_feat = stored_cf["full_report"].iloc[0]["spearman_r"]

    assert np.isfinite(r_const_feat)
    assert r_const_feat == pytest.approx(0.0, abs=1e-9)
    assert res_const_feat["kept_features_count"] == 0

    # 7b. All-identical labels test: feature_matrix = [[1], [0], [0]], labels = ["catA", "catA", "catA"]
    session_ident_labels = SessionState()
    session_ident_labels.feature_matrix = sp.csr_matrix([[1], [0], [0]], dtype=int)
    session_ident_labels.feature_names = ["var_feature"]
    session_ident_labels.set_labels(np.array(["catA", "catA", "catA"]))

    tools_ident_labels = make_filtering_tools(session_ident_labels)
    spearman_ident_labels = next(
        t for t in tools_ident_labels if t.name == "spearman_filter_tool"
    )

    res_ident_labels = spearman_ident_labels.invoke({"target_class": "catA", "threshold": 0.0})
    stored_il = session_ident_labels.results_store[res_ident_labels["result_id"]]
    r_ident_labels = stored_il["full_report"].iloc[0]["spearman_r"]

    assert np.isfinite(r_ident_labels)
    assert r_ident_labels == pytest.approx(0.0, abs=1e-9)
    assert res_ident_labels["kept_features_count"] == 0

    # 8. Boundary test: Perfectly correlated feature [1, 0, 0] with ["catB", "catA", "catA"]
    session_perf = SessionState()
    session_perf.feature_matrix = sp.csr_matrix([[1], [0], [0]], dtype=int)
    session_perf.feature_names = ["perf_feature"]
    session_perf.set_labels(np.array(["catB", "catA", "catA"]))

    tools_perf = make_filtering_tools(session_perf)
    spearman_perf = next(
        t for t in tools_perf if t.name == "spearman_filter_tool"
    )

    result_perf_default = spearman_perf.invoke({"target_class": "catB", "threshold": 0.0})
    stored_perf = session_perf.results_store[result_perf_default["result_id"]]
    perf_report = stored_perf["full_report"]
    actual_perf_r = perf_report.iloc[0]["spearman_r"]

    assert -1.0 <= actual_perf_r <= 1.0
    assert actual_perf_r == pytest.approx(1.0, abs=1e-9)

    # With threshold=1.0, r > threshold (1.0 > 1.0) is False -> keeps 0 features
    result_perf_thresh1 = spearman_perf.invoke({"target_class": "catB", "threshold": 1.0})
    assert result_perf_thresh1["kept_features_count"] == 0
    assert session_perf.artifacts["spearman_filter"] == []
