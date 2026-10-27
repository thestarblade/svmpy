# svmpy

A tiny, dependency-free binary SVM in Python, ported from
[svmjs](https://github.com/karpathy/svmjs). Implements the simplified SMO
algorithm with linear, RBF, and custom kernels.

## Usage

```python
from svmpy import SVM

data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [-1, 1, 1, -1]

svm = SVM()
svm.train(data, labels, kernel="rbf", rbfsigma=0.5)

svm.predict(data)        # [-1, 1, 1, -1]
svm.margins(data)        # raw decision values
svm.margin_one(data[0])  # single margin
```

`train` takes keyword-only options: `C=1.0`, `tol=1e-4`, `alphatol=1e-7`,
`maxiter=10000`, `numpasses=10`, `kernel="linear"`, `rbfsigma=0.5`,
`memoize=False`. `kernel` may be `"linear"`, `"rbf"`, or any callable
`K(v1, v2)`.

Use `SVM(rng=random.Random(42))` for reproducible training. Predicting before
training raises `NotFittedError`.

## Save / load

```python
import json
from svmpy import SVM

payload = svm.to_dict()
svm2 = SVM.from_dict(payload)
svm3 = SVM.from_dict(json.loads(json.dumps(payload)))
```

Field names match svmjs. Custom kernels cannot be serialized.

## Tests

```bash
python3 -m unittest discover -v
```

MIT License. Original implementation (c) 2012 Andrej Karpathy.
