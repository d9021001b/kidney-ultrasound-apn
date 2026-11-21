import numpy as np

from nckuh_dmsa_inductive_graph import diffuse, edges


def test_edges_exclude_same_patient_and_duplicate_neighbors():
    x = np.array([[0.], [1.], [2.], [3.], [4.]])
    patients = np.array(["a", "a", "b", "b", "c"])
    graph = edges(x, x, patients, patients, k=2)
    np.testing.assert_array_equal(np.flatnonzero(graph[0]), [2, 4])
    assert not graph[2, 3]
    np.testing.assert_allclose(graph.sum(1), 1)
    np.testing.assert_allclose(diffuse(graph, np.ones(5)), 1)
