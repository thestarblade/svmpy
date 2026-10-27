"""svmpy: a lightweight binary SVM trained with the SMO algorithm.

Python port of Andrej Karpathy's svmjs, with the algorithm logic following
``svmjs/lib/svm.js``.
"""

from .svm import NotFittedError, SVM, linear_kernel, make_rbf_kernel

__version__ = "0.1.0"
__all__ = ["SVM", "NotFittedError", "linear_kernel", "make_rbf_kernel"]
