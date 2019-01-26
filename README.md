# pyoptree
Python Optimal Tree

### Install 
#### First install pyoptree through pip
```
pip3 install pyoptree
```

#### Then install solver (IMPORTANT!) 
The user needs to have **IBM Cplex** or **Gurobi** installed on their computer, and make sure that **the executable has been added to PATH environment variable** (i.e. command `cplex` or `gurobi` can be run on terminal). 

### Example 
```python
import pandas as pd
from pyoptree.optree import OptimalHyperTreeModel, OptimalTreeModel

data = pd.DataFrame({
        "index": ['A', 'C', 'D', 'E', 'F'],
        "x1": [1, 2, 2, 2, 3],
        "x2": [1, 2, 1, 0, 1],
        "y": [1, 1, -1, -1, -1]
    })
test_data = pd.DataFrame({
    "index": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    "x1": [1, 1, 2, 2, 2, 3, 3],
    "x2": [1, 2, 2, 1, 0, 1, 0],
    "y": [1, 1, 1, -1, -1, -1, -1]
})
model = OptimalHyperTreeModel(["x1", "x2"], "y", tree_depth=2, N_min=1, alpha=0.1, solver_name="cplex")
model.train(data)

print(model.predict(test_data))
```

### Todos 
1. Implement "Warm Start" to speed up the time to solve the Mixed Integer Linear Programming (MILP); 
2. Implement heuristics such as Generic Algorithms to approximate the optimal solution rapidly (but may be loss of accuracy);

### Hyper-parameter Tuning Experience 
Generally, there are three hyper-parameters that could be tuned for the Optimal Tree model: 
- **tree_depth**: the depth of the tree. 
- **N_min**: minimum number of samples in each leaf node if that node contains any sample. 
- **alpha**: coefficient of the regularization term. 

As experimenting with the Iris dataset on a 4-thread computer using Cplex 12.8, the results are listed as follows: 

| 参数名称 | 参数值 | 求解消耗时间 | 其他参数设置 | 目标函数最优值 |
| ------ | ------ | ------ |  ------  | ------ |
| N_min | 30 | 20.80 sec | tree_depth = 2, alpha = 0.1   | 3.5 | 
| N_min | 10 | 3027.00 sec |  tree_depth = 2, alpha = 0.1  | 3.5 | 

From the results, we can get the following insights: 
1. **N_min** has great impact on the solving time (the larger N_min, the quicker). But a larger N_min may cause the problem
 infeasible. So care must be taken when tuning N_min. 

More experiments' results would be added in the future. 