"""Tests for svmpy.

Mirrors svmjs/test/testnode.js and adds coverage for the Python-facing API.

Run from the svmpy directory with::

    python3 -m unittest discover -v
"""

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


if __name__ == "__main__":
    unittest.main()
