import unittest

import numpy as np

from run_uniusnet_frozen_development import final_token_selection


class UniUSNetPatchMappingTests(unittest.TestCase):
    def test_full_mask_selects_all_final_tokens(self):
        selected = final_token_selection(np.ones(196, dtype=bool))
        self.assertEqual(selected.shape, (49,))
        self.assertEqual(int(selected.sum()), 49)

    def test_single_fine_patch_has_deterministic_fallback(self):
        fine = np.zeros(196, dtype=bool)
        fine[0] = True
        selected = final_token_selection(fine)
        self.assertEqual(int(selected.sum()), 1)
        self.assertTrue(selected[0])

    def test_two_of_four_fine_patches_select_one_final_token(self):
        fine = np.zeros((14, 14), dtype=bool)
        fine[4:6, 8] = True
        selected = final_token_selection(fine.reshape(-1))
        self.assertEqual(int(selected.sum()), 1)
        self.assertTrue(selected.reshape(7, 7)[2, 4])


if __name__ == "__main__":
    unittest.main()
