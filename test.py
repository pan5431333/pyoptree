import pandas as pd
from sklearn.datasets import load_iris
from pyoptree.optree import OptimalHyperTreeModel, OptimalTreeModel

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

features, label = load_iris(True)
data = pd.DataFrame(data=features, columns=["x1", "x2", "x3", "x4"])
data["label"] = label

model = OptimalHyperTreeModel(["x1", "x2", "x3", "x4"], "label", tree_depth=2, N_min=30, alpha=0.1, solver_name="cplex")
model.train(data)

print(model.predict(data))
print(model.get_feature_importance())
