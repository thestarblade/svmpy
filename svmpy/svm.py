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

__all__ = ["linear_kernel", "make_rbf_kernel"]


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
