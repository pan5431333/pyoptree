# pyoptree
Python Implementation of Bertsimas's "Optimal classification trees". 

### Dependencies 
#### Python dependencies (through pip)
1. numpy 
2. pandas 
3. pyomo 

#### Solver dependencies 
The user needs to have IBM Cplex or Gurobi installed on their computer, and make sure that the executable has been added to PATH environment viariable (i.e. command `cplex` or `gurobi` can be run on terminal). 

### Example 
```python
import pandas as pd
from optree.optree import OptimalHyperTreeModel, OptimalTreeModel

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