"""svmpy: a lightweight binary SVM trained with the SMO algorithm."""

from .svm import SVM, linear_kernel, make_rbf_kernel

__version__ = "0.1.0"
__all__ = ["SVM", "linear_kernel", "make_rbf_kernel"]
