import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool_known_answer_1():
    """Test load_dataset_tool against TEST_FIXTURE.md Known Answer #1.

    Asserts that loading 'agent_dev/sample_fixture.csv' yields:
    - Exactly 8 total documents (n_documents == 8)
    - counts_per_category == {'catA': 4, 'catB': 4}
    """
    # 1. Instantiate the single session for state tracking
    session = SessionState()

    # 2. Load premade tools from 'premade_tools' directory
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # 3. Invoke load_dataset_tool with the fixture CSV path and column mapping
    result = load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 4. Assert against TEST_FIXTURE.md Known Answer #1
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}

    # 5. Verify that session.dataframe was correctly populated
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
