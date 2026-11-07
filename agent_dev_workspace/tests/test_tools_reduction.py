import json
import pytest
import numpy as np
import sklearn
import scipy
import platform
from sklearn.manifold import TSNE
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def _prepare_session():
    """Helper to initialize a fresh session, populate DTM, and verify fixture data."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"ngram_range": [1, 1]})

    # Verify DTM matches fixture 8x5 vocabulary & matrix
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab

    dtm_dense = session.feature_matrix.toarray()
    expected_dtm_dense = np.array([
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ])
    assert np.array_equal(dtm_dense, expected_dtm_dense)

    return session, dtm_dense


def test_reduce_dimensions_invalid_n_components():
    session, _ = _prepare_session()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    with pytest.raises(ValueError, match="n_components must be at least 2"):
        reduce_dimensions_tool.invoke({"method": "pca", "n_components": 1})


def test_reduce_dimensions_pca():
    session, _ = _prepare_session()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    session.pending_figure = None
    pca_res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})
    assert "error" not in pca_res

    expected_pca_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    expected_pca_var_ratio = [0.7532, 0.1965]

    pca_coords = np.array(pca_res["reduced_coordinates"])
    assert np.allclose(pca_coords, expected_pca_coords, rtol=0, atol=1e-3)
    assert np.allclose(pca_res["explained_variance_ratio"], expected_pca_var_ratio, rtol=0, atol=1e-3)

    assert session.pending_figure is not None
    assert np.allclose(session.artifacts["reduced_coords_pca"], pca_coords, rtol=0, atol=1e-3)

    result_id = pca_res["result_id"]
    record = session.results_store[result_id]["record"]
    assert record.args["method"] == "pca"
    assert record.args["n_components"] == 2

    json_str = json.dumps(record.summary)
    assert isinstance(json_str, str)


def test_reduce_dimensions_tsne():
    session, dtm_dense = _prepare_session()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    session.pending_figure = None
    tsne_res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2.0,
        "random_state": 42
    })
    assert "error" not in tsne_res

    expected_tsne_coords = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973],
    ])

    tsne_coords = np.array(tsne_res["reduced_coordinates"])

    # Direct TSNE fit on same matrix for diagnostic check
    tsne_direct = TSNE(n_components=2, perplexity=2.0, random_state=42)
    direct_coords = tsne_direct.fit_transform(dtm_dense)
    max_diff_with_tool = np.max(np.abs(tsne_coords - direct_coords))

    # Grid search across 6 combinations (dtypes x TSNE methods)
    dtypes = [np.int64, np.float32, np.float64]
    methods = ["barnes_hut", "exact"]
    combo_results = []

    for dt in dtypes:
        for m in methods:
            matrix_casted = dtm_dense.astype(dt)
            model = TSNE(n_components=2, perplexity=2.0, random_state=42, method=m)
            coords = model.fit_transform(matrix_casted)
            diff = np.max(np.abs(coords - expected_tsne_coords))
            kl_div = getattr(model, "kl_divergence_", None)
            combo_results.append({
                "dtype": str(dt.__name__),
                "method": m,
                "first_coord": coords[0].tolist(),
                "max_diff_vs_expected": float(diff),
                "kl_divergence": float(kl_div) if kl_div is not None else None
            })

    env_info = (
        f"platform={platform.platform()}, "
        f"python={platform.python_version()}, "
        f"numpy={np.__version__}, "
        f"scipy={scipy.__version__}, "
        f"sklearn={sklearn.__version__}"
    )

    diag_msg = (
        f"env=[{env_info}], "
        f"DTM_dtype={dtm_dense.dtype}, "
        f"max_diff_tool_vs_direct={max_diff_with_tool}, "
        f"direct_coords={direct_coords.tolist()}, "
        f"combinations={combo_results}"
    )

    assert np.allclose(tsne_coords, expected_tsne_coords, rtol=0, atol=1e-3), diag_msg

    assert session.pending_figure is not None
    assert np.allclose(session.artifacts["reduced_coords_tsne"], tsne_coords, rtol=0, atol=1e-3)

    tsne_record = session.results_store[tsne_res["result_id"]]["record"]
    assert tsne_record.args["method"] == "tsne"
    assert tsne_record.args["n_components"] == 2
    assert tsne_record.args["perplexity"] == 2.0
    assert tsne_record.args["random_state"] == 42
    json.dumps(tsne_record.summary)


def test_reduce_dimensions_umap():
    session, _ = _prepare_session()
    reduction_tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    session.pending_figure = None
    umap_res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })
    assert "error" not in umap_res

    expected_umap_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])

    umap_coords = np.array(umap_res["reduced_coordinates"])
    assert np.allclose(umap_coords, expected_umap_coords, rtol=0, atol=1e-3)

    assert session.pending_figure is not None
    assert np.allclose(session.artifacts["reduced_coords_umap"], umap_coords, rtol=0, atol=1e-3)

    umap_record = session.results_store[umap_res["result_id"]]["record"]
    assert umap_record.args["method"] == "umap"
    assert umap_record.args["n_components"] == 2
    assert umap_record.args["n_neighbors"] == 3
    assert umap_record.args["random_state"] == 42
    json.dumps(umap_record.summary)
