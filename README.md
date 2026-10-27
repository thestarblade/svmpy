# svmpy

A tiny, dependency-free binary SVM in Python, ported from
[svmjs](https://github.com/karpathy/svmjs).

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

Use `SVM(rng=random.Random(42))` for reproducible training.

## Save / load

```python
import json
from svmpy import SVM

payload = svm.to_dict()
svm2 = SVM.from_dict(payload)
svm3 = SVM.from_dict(json.loads(json.dumps(payload)))
```

Field names match svmjs. Custom kernels cannot be serialized.

MIT License. Original implementation (c) 2012 Andrej Karpathy.
