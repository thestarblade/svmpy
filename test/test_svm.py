"""Tests for svmpy.

Mirrors svmjs/test/testnode.js and adds coverage for the Python-facing API.

Run from the svmpy directory with::

    python3 -m unittest discover -v
"""

import json
import unittest

import svmpy


XOR_DATA = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_LABELS = [-1, 1, 1, -1]

LINEAR_DATA = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
LINEAR_LABELS = [-1, -1, 1, 1]


class SVMPortTest(unittest.TestCase):

    def test_rbf_xor(self):
        svm = svmpy.SVM()
        stats = svm.train(XOR_DATA, XOR_LABELS, kernel="rbf", rbfsigma=0.5)

        self.assertGreater(stats["iters"], 0)
        self.assertEqual(svm.predict(XOR_DATA), XOR_LABELS)
        self.assertEqual(svm.n_samples, 4)
        self.assertEqual(svm.n_features, 2)
        self.assertIsNone(svm.w)

    def test_linear_kernel(self):
        svm = svmpy.SVM()
        svm.train(LINEAR_DATA, LINEAR_LABELS, kernel="linear", C=1.0)

        self.assertIsNotNone(svm.w)
        self.assertEqual(len(svm.w), 2)
        self.assertEqual(svm.predict([[0.5, 0.5], [2.5, 2.5]]), [-1, 1])

    def test_dict_roundtrip_rbf(self):
        svm = svmpy.SVM()
        svm.train(XOR_DATA, XOR_LABELS, kernel="rbf", rbfsigma=0.5)

        payload = svm.to_dict()
        svm2 = svmpy.SVM.from_dict(payload)

        self.assertEqual(svm2.predict(XOR_DATA), XOR_LABELS)
        self.assertEqual(svm2.n_samples, svm.n_samples)
        self.assertEqual(json.loads(json.dumps(payload))["kernelType"], "rbf")

    def test_dict_roundtrip_linear(self):
        svm = svmpy.SVM()
        svm.train(LINEAR_DATA, LINEAR_LABELS, kernel="linear")

        payload = svm.to_dict()
        svm2 = svmpy.SVM.from_dict(payload)

        self.assertEqual(svm2.predict([[0.5, 0.5], [2.5, 2.5]]), [-1, 1])
        self.assertEqual(svm2.w, svm.w)
        self.assertEqual(svm2.b, svm.b)

    def test_custom_kernel_cannot_be_serialized(self):
        svm = svmpy.SVM()
        svm.train(
            [[0.0], [1.0]],
            [-1, 1],
            kernel=svmpy.linear_kernel,
        )
        self.assertEqual(svm.kernel_type, "custom")

        with self.assertRaises(TypeError):
            svm.to_dict()

    def test_unknown_kernel_type_in_payload(self):
        payload = {"N": 1, "D": 1, "b": 0.0, "kernelType": "poly"}

        with self.assertRaises(ValueError):
            svmpy.SVM.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
