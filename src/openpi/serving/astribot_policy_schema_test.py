import numpy as np

from openpi.serving import astribot_policy_schema as schema


def test_vendored_schema_matches_orin_contract() -> None:
    assert schema.POLICY_PROTOCOL_VERSION == 1
    assert schema.POLICY_SCHEMA_HASH == "4f7eda439cd87c448c483d36fa91a8ada58158d9d2947d6c28b0b7e0459b21e7"
    assert len(schema.CANONICAL_BODY_JOINTS) == 23
    assert len(schema.HAND_CONTROL_JOINTS) == 16
    assert len(schema.HAND_STATE_JOINTS) == 21
    assert schema.ACTION_DIM == 55
    assert schema.CANONICAL_ACTION_JOINTS == schema.ACTION_JOINT_NAMES


def test_schema_reorders_named_vectors_and_selects_hand_driver_controls() -> None:
    body_values = np.arange(23, dtype=np.float64)
    received_body_names = tuple(reversed(schema.CANONICAL_BODY_JOINTS))
    received_body_values = body_values[::-1]
    assert np.array_equal(
        schema.canonical_vector(
            received_body_names,
            received_body_values,
            schema.CANONICAL_BODY_JOINTS,
            field="body",
        ),
        body_values,
    )

    hand_values = np.arange(21, dtype=np.float64) + 100.0
    received_hand_names = tuple(reversed(schema.HAND_STATE_JOINTS))
    received_hand_values = hand_values[::-1]
    selected = schema.hand_control_from_state(
        received_hand_names,
        received_hand_values,
        field="hand",
    )
    expected = hand_values[[schema.HAND_STATE_JOINTS.index(name) for name in schema.HAND_CONTROL_JOINTS]]
    assert np.array_equal(selected, expected)


def test_single_action_response_validator_matches_stream_contract() -> None:
    mask = schema.enabled_action_mask(
        control_scope="upper_body",
        body=True,
        left_hand=True,
        right_hand=False,
    )
    response = {
        "type": "action",
        "protocol_version": schema.POLICY_PROTOCOL_VERSION,
        "schema_hash": schema.POLICY_SCHEMA_HASH,
        "session_id": "test-session",
        "action_sequence": 3,
        "source_observation_sequence": 9,
        "body_joint_names": list(schema.CANONICAL_BODY_JOINTS),
        "hand_control_joint_names": list(schema.HAND_CONTROL_JOINTS),
        "enabled_action_mask": mask,
        "action": np.zeros(schema.ACTION_DIM, dtype=np.float32),
    }
    action, action_sequence, source_sequence = schema.validate_single_action_response(
        response,
        session_id="test-session",
        expected_mask=mask,
    )
    assert action.shape == (schema.ACTION_DIM,)
    assert action_sequence == 3
    assert source_sequence == 9
