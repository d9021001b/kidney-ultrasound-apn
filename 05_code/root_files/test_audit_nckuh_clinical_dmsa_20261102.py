import unittest
from datetime import datetime

from audit_nckuh_clinical_dmsa_20261102 import MAP, code, date_value, patient_key


class ClinicalReferenceTests(unittest.TestCase):
    def test_laterality_and_blank(self):
        self.assertEqual(MAP[code("r")], (0, 1))
        self.assertEqual(MAP[code("L")], (1, 0))
        self.assertEqual(MAP[code("bil")], (1, 1))
        self.assertEqual(MAP[code("N")], (0, 0))
        self.assertIsNone(code(None))
        self.assertIsNone(code(" "))
        with self.assertRaises(ValueError):
            code("unknown")

    def test_identifier_normalization_rejects_bad_keys(self):
        self.assertEqual(patient_key("00123456"), "123456")
        self.assertEqual(patient_key(123456.0), "123456")
        with self.assertRaises(ValueError):
            patient_key(123456.5)

    def test_malformed_exam_date_is_not_guessed(self):
        epoch = datetime(1899, 12, 30)
        self.assertEqual(date_value("1010808", epoch), "2012-08-08")
        self.assertIsNone(date_value("201208087", epoch))


if __name__ == "__main__":
    unittest.main()
