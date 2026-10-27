# MIT License
#
# Original JavaScript implementation:
#   Copyright (c) 2012 Andrej Karpathy (https://github.com/karpathy/svmjs)
# Python port:
#   Algorithm logic follows svmjs/lib/svm.js.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""A binary SVM trained with the SMO algorithm.

This is a dependency-free Python port of Andrej Karpathy's svmjs.  The SMO
update rules, stopping criterion and kernel handling follow
``svmjs/lib/svm.js``.  Data is a list of N D-dimensional vectors and labels is
a list of N values, each either 1 or -1.

Reference: "The Simplified SMO Algorithm"
(http://math.unt.edu/~hsp0009/smo.pdf)

Simple usage::

    from svmpy import SVM

    svm = SVM()
    svm.train(data, labels)
    predictions = svm.predict(test_data)
"""

import math
import random

__all__ = ["SVM", "linear_kernel", "make_rbf_kernel"]


class SVM:
    """Binary support vector machine trained with simplified SMO."""

    def __init__(self):
        self.n_samples = 0
        self.n_features = 0
        self.b = 0.0
        self.alpha = []
        self.data = []
        self.labels = []
        self.kernel = linear_kernel
        self.kernel_type = "linear"
        self.w = None
        self.rbf_sigma = None

    def train(self, data, labels, *, C=1.0, tol=1e-4, alphatol=1e-7,
              maxiter=10000, numpasses=10, kernel="linear",
              rbfsigma=0.5):
        """Train the SVM.

        ``data`` is an N x D list of floats, ``labels`` contains 1 or -1.
        All options are keyword-only.  ``kernel`` may be ``"linear"``,
        ``"rbf"`` or any callable ``K(v1, v2)``.

        A dict of options can still be applied with
        ``svm.train(data, labels, **options)``.
        """
        # instantiate kernel. It can be given as a string or as a callable.
        kernel_fn, kernel_type, rbf_sigma = _resolve_kernel(kernel, rbfsigma)

        self.kernel = kernel_fn
        self.kernel_type = kernel_type
        self.rbf_sigma = rbf_sigma

        # we need these in helper functions
        self.data = data
        self.labels = labels

        # initializations
        self.n_samples = len(data)
        self.n_features = len(data[0])
        self.alpha = [0.0] * self.n_samples
        self.b = 0.0
        self.w = None

        # run SMO algorithm
        iters = 0
        passes = 0
        while passes < numpasses and iters < maxiter:
            alpha_changed = 0
            for i, (xi, yi) in enumerate(zip(data, labels)):
                Ei = self._margin_one(xi) - yi
                if ((yi * Ei < -tol and self.alpha[i] < C)
                        or (yi * Ei > tol and self.alpha[i] > 0)):

                    # alpha_i needs updating! Pick a j != i uniformly.
                    j = random.randrange(self.n_samples - 1)
                    if j >= i:
                        j += 1

                    xj = data[j]
                    yj = labels[j]
                    Ej = self._margin_one(xj) - yj

                    # calculate L and H bounds for j to ensure we're in
                    # [0 C] x [0 C] box
                    ai = self.alpha[i]
                    aj = self.alpha[j]
                    if yi == yj:
                        L = max(0.0, ai + aj - C)
                        H = min(C, ai + aj)
                    else:
                        L = max(0.0, aj - ai)
                        H = min(C, C + aj - ai)

                    if abs(L - H) < 1e-4:
                        continue

                    eta = (2 * self._kernel_result(i, j)
                           - self._kernel_result(i, i)
                           - self._kernel_result(j, j))
                    if eta >= 0:
                        continue

                    # compute new alpha_j and clip it, then derive alpha_i
                    newaj = aj - yj * (Ei - Ej) / eta
                    newaj = min(max(newaj, L), H)
                    if abs(aj - newaj) < 1e-4:
                        continue
                    self.alpha[j] = newaj
                    newai = ai + yi * yj * (aj - newaj)
                    self.alpha[i] = newai

                    # update the bias term
                    b1 = (self.b - Ei
                          - yi * (newai - ai) * self._kernel_result(i, i)
                          - yj * (newaj - aj) * self._kernel_result(i, j))
                    b2 = (self.b - Ej
                          - yi * (newai - ai) * self._kernel_result(i, j)
                          - yj * (newaj - aj) * self._kernel_result(j, j))
                    self.b = 0.5 * (b1 + b2)
                    if 0 < newai < C:
                        self.b = b1
                    if 0 < newaj < C:
                        self.b = b2

                    alpha_changed += 1

            iters += 1
            if not alpha_changed:
                passes += 1
            else:
                passes = 0

        return {"iters": iters}

    def _margin_one(self, inst):
        f = self.b
        f += sum(a * y * self.kernel(inst, sv)
                 for a, y, sv in zip(self.alpha, self.labels, self.data))
        return f

    def margin_one(self, inst):
        """Return the margin of one D-dimensional instance."""
        return self._margin_one(inst)

    def predict_one(self, inst):
        return 1 if self.margin_one(inst) > 0 else -1

    def margins(self, data):
        """Return a list of margins for an N x D list of instances."""
        return [self._margin_one(inst) for inst in data]

    def _kernel_result(self, i, j):
        return self.kernel(self.data[i], self.data[j])

    def predict(self, data):
        """Return a list of 1 or -1 predictions for an N x D list."""
        return [1 if m > 0 else -1 for m in self.margins(data)]


# Kernels

def make_rbf_kernel(sigma):
    denom = 2.0 * sigma * sigma

    def kernel(v1, v2):
        return math.exp(
            -sum((a - b) ** 2 for a, b in zip(v1, v2)) / denom
        )

    return kernel


def linear_kernel(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


def _resolve_kernel(kernel, rbfsigma):
    """Return ``(kernel_fn, kernel_type, rbf_sigma)``."""
    if isinstance(kernel, str):
        if kernel == "linear":
            return linear_kernel, "linear", None
        if kernel == "rbf":
            if rbfsigma <= 0:
                raise ValueError("rbfsigma must be positive")
            return make_rbf_kernel(rbfsigma), "rbf", rbfsigma
        raise ValueError(
            f"unknown kernel {kernel!r}; expected 'linear', 'rbf' or a callable"
        )
    if not callable(kernel):
        raise TypeError("kernel must be 'linear', 'rbf' or a callable")
    return kernel, "custom", None
