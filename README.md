# pyoptree
Python Optimal Tree

There is a distributional version of this algorithm running on Spark called [optree4s](https://github.com/pan5431333/optree4s). Note that in order to scale the algorithm to "big data", only Local Search method is implemented in optree4s.
 
Both of this Python version and the Scala/Spark version are under active(?) development. But since my work in Alibaba is so busy, I certainly welcome anyone's fork&pull request. 


### Install 
#### First install pyoptree through pip
```
pip3 install pyoptree
```

#### Then install solver (IMPORTANT!) 
The user needs to have **IBM Cplex** or **Gurobi** installed on their computer, and make sure that **the executable has been added to PATH environment variable** (i.e. command `cplex` or `gurobi` can be run on terminal). 

### Example 

A minimal runnable example is as follows, please don't hesitate to contact me by (meng.pan95@gmail.com) if you encountered any trouble or have any suggestion. I usually will check my Email every night, and I promise to respond every Email from GitHub~~

```python
import pandas as pd
from pyoptree.optree import OptimalHyperTreeModel, OptimalTreeModel

data = pd.DataFrame({
    "index": ['A', 'C', 'D', 'E', 'F'],
    "x1": [1, 2, 2, 2, 3],
    "x2": [1, 2, 1, 0, 1],
    "y": [1, 1, 0, 0, 0]
})
test_data = pd.DataFrame({
    "index": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    "x1": [1, 1, 2, 2, 2, 3, 3],
    "x2": [1, 2, 2, 1, 0, 1, 0],
    "y": [1, 1, 1, 0, 0, 0, 0]
})
model = OptimalHyperTreeModel(["x1", "x2"], "y", tree_depth=2, N_min=1, alpha=0.1, solver_name="cplex")
model.train(data, train_method="mio")

print(model.predict(test_data))
```

### Todos 
1. Use the solution from the previous depth tree as a "Warm Start" to speed up the time to solve the Mixed Integer Linear Programming (MILP); （Done √）
2. Use the solution from sklearn's CART to give a good initial solution (Done √);
