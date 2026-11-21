import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_uri_multiview_graph_20261107 as study


def test_patient_slots_follow_anatomic_view_order():
    patients = pd.DataFrame({"patient_alias": ["P1", "P2"]})
    images = pd.DataFrame(
        {
            "patient_alias": ["P1", "P1", "P2"],
            "view": ["RT", "LL", "LT"],
        }
    )
    values = np.asarray([[1, 1], [2, 2], [3, 3]], np.float32)
    slots, mask = study.make_patient_slots(values, images, patients)
    np.testing.assert_array_equal(mask, [[True, False, False, True], [False, True, False, False]])
    np.testing.assert_allclose(slots[0, 0], [2, 2])
    np.testing.assert_allclose(slots[0, 3], [1, 1])


def test_relation_and_disconnected_control_have_equal_capacity():
    disconnected = study.MultiviewRelationGraph(False)
    relation = study.MultiviewRelationGraph(True)
    assert study.parameter_count(disconnected) == study.parameter_count(relation)


def test_graph_messages_use_contralateral_node_but_control_does_not():
    torch.manual_seed(7)
    disconnected = study.MultiviewRelationGraph(False, input_dim=2, hidden_dim=4).eval()
    relation = study.MultiviewRelationGraph(True, input_dim=2, hidden_dim=4).eval()
    relation.load_state_dict(disconnected.state_dict())
    mask = torch.ones((1, 4), dtype=torch.bool)
    original = torch.zeros((1, 4, 2))
    changed = original.clone()
    changed[0, 2] = torch.tensor([3.0, -2.0])
    with torch.inference_mode():
        control_a = disconnected.message_nodes(original, mask)[0, 0]
        control_b = disconnected.message_nodes(changed, mask)[0, 0]
        graph_a = relation.message_nodes(original, mask)[0, 0]
        graph_b = relation.message_nodes(changed, mask)[0, 0]
    torch.testing.assert_close(control_a, control_b)
    assert not torch.allclose(graph_a, graph_b)


def test_missing_view_forward_is_finite():
    model = study.MultiviewRelationGraph(True)
    values = torch.randn(3, 4, 32)
    mask = torch.tensor(
        [[True, True, True, True], [True, False, True, False], [True, True, False, False]]
    )
    result = model(values, mask)
    assert result.shape == (3, 2)
    assert torch.isfinite(result).all()
