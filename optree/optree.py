import pandas as pd
from pyomo.environ import *  # !! Please don't delete this
from pyomo.core.base.PyomoModel import *
from pyomo.core.base.constraint import *
from pyomo.core.base.objective import *
from pyomo.core.base.var import *
from pyomo.core.kernel.set_types import *
from pyomo.opt.base.solvers import *
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', )


class OptimalTreeModel:
    def __init__(self, x_cols: list, y_col: str, tree_depth: int, N_min: int, alpha: float = 0.01,
                 M: int = 1e6, epsilon: float = 1e-4,
                 solver_name: str = "cplex"):
        self.y_col = y_col
        self.P = len(x_cols)
        self.P_range = x_cols
        self.K_range = None
        self.solver_name = solver_name
        self.D = tree_depth
        self.Nmin = N_min
        self.M = M
        self.epsilon = epsilon
        self.alpha = alpha
        self.is_trained = False
        nodes = list(range(1, (2 ** (self.D + 1))))
        self.parent_nodes = nodes[0: 2 ** (self.D + 1) - 2 ** self.D - 1]
        self.leaf_ndoes = nodes[-2 ** self.D:]
        self.normalizer = {}

        # solutions
        self.l = None
        self.c = None
        self.d = None
        self.a = None
        self.b = None
        self.Nt = None
        self.Nkt = None
        self.Lt = None

        assert tree_depth > 0, "Tree depth must be greater than 0! (Actual: {0})".format(tree_depth)

    def train(self, data: pd.DataFrame, show_training_process: bool = True):
        data = data.copy()
        for col in self.P_range:
            col_max = max(data[col])
            col_min = min(data[col])
            self.normalizer[col] = (col_max, col_min)
            if col_max != col_min:
                data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                data[col] = 1
        model = self.__generate_model(data.reset_index(drop=True))
        solver = SolverFactory(self.solver_name)
        res = solver.solve(model, tee=show_training_process)
        status = str(res.solver.termination_condition)
        self.is_trained = True
        loss = value(model.obj)
        self.l = {t: value(model.l[t]) for t in self.leaf_ndoes}
        self.c = {t: [value(model.c[k, t]) for k in self.K_range] for t in self.leaf_ndoes}
        self.d = {t: value(model.d[t]) for t in self.parent_nodes}
        self.a = {t: [value(model.a[j, t]) for j in self.P_range] for t in self.parent_nodes}
        self.b = {t: value(model.bt[t]) for t in self.parent_nodes}
        self.Nt = {t: value(model.Nt[t]) for t in self.leaf_ndoes}
        self.Nkt = {t: [value(model.Nkt[k, t]) for k in self.K_range] for t in self.leaf_ndoes}
        self.Lt = {t: value(model.Lt[t]) for t in self.leaf_ndoes}
        logging.info("Training done. Loss: {1}. Optimization status: {0}".format(status, loss))

    def predict(self, data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet! Please use `train()` to train the model first!")

        new_data = data.copy()
        new_data_cols = data.columns
        for col in self.P_range:
            if col not in new_data_cols:
                raise ValueError("Column {0} is not in the given data for prediction! ".format(col))
            col_max, col_min = self.normalizer[col]
            if col_max != col_min:
                new_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                new_data[col] = 1

        prediction = []
        for j in range(new_data.shape[0]):
            x = np.array([new_data.ix[j, i] for i in self.P_range])
            t = 1
            d = 0
            while d < self.D:
                at = np.array(self.a[t])
                bt = self.b[t]
                if at.dot(x) < bt:
                    t = t * 2
                else:
                    t = t * 2 + 1
                d = d + 1
            y_hat = self.c[t]
            prediction.append(self.K_range[y_hat.index(max(y_hat))])
        data["prediction"] = prediction
        return data

    def __generate_model(self, data: pd.DataFrame):
        model = ConcreteModel(name="OptimalTreeModel")
        n = data.shape[0]
        label = data[[self.y_col]]
        label["__value__"] = 1
        Y = label.pivot(columns=self.y_col, values="__value__")

        L_hat = max(label.groupby(by=self.y_col).sum()["__value__"]) / n

        Y.fillna(value=-1, inplace=True)

        n_range = range(n)
        K_range = sorted(list(set(data[self.y_col])))
        P_range = self.P_range

        self.K_range = K_range

        parent_nodes = self.parent_nodes
        leaf_ndoes = self.leaf_ndoes

        # Variables
        model.z = Var(n_range, leaf_ndoes, within=Binary)
        model.l = Var(leaf_ndoes, within=Binary)
        model.c = Var(K_range, leaf_ndoes, within=Binary)
        model.d = Var(parent_nodes, within=Binary)
        model.a = Var(P_range, parent_nodes, within=Binary)

        model.Nt = Var(leaf_ndoes, within=NonNegativeReals)
        model.Nkt = Var(K_range, leaf_ndoes, within=NonNegativeReals)
        model.Lt = Var(leaf_ndoes, within=NonNegativeReals)
        model.bt = Var(parent_nodes, within=NonNegativeReals)

        # Constraints
        model.integer_relationship_constraints = ConstraintList()
        for t in leaf_ndoes:
            model.integer_relationship_constraints.add(
                expr=sum([model.c[k, t] for k in K_range]) == model.l[t]
            )
        for i in n_range:
            for t in leaf_ndoes:
                model.integer_relationship_constraints.add(
                    expr=model.z[i, t] <= model.l[t]
                )
        for i in n_range:
            model.integer_relationship_constraints.add(
                expr=sum([model.z[i, t] for t in leaf_ndoes]) == 1
            )
        for t in leaf_ndoes:
            model.integer_relationship_constraints.add(
                expr=sum([model.z[i, t] for i in n_range]) >= model.l[t] * self.Nmin
            )
        for t in parent_nodes:
            model.integer_relationship_constraints.add(
                expr=sum([model.a[j, t] for j in P_range]) == model.d[t]
            )
        for t in parent_nodes:
            if t != 1:
                model.integer_relationship_constraints.add(
                    expr=model.d[t] <= model.d[self.__parent(t)]
                )

        model.leaf_samples_constraints = ConstraintList()
        for t in leaf_ndoes:
            model.leaf_samples_constraints.add(
                expr=model.Nt[t] == sum([model.z[i, t] for i in n_range])
            )
        for t in leaf_ndoes:
            for k in K_range:
                model.leaf_samples_constraints.add(
                    expr=model.Nkt[k, t] == sum([model.z[i, t] * (1 + Y.loc[i, k]) / 2.0 for i in n_range])
                )

        model.leaf_error_constraints = ConstraintList()
        for k in K_range:
            for t in leaf_ndoes:
                model.leaf_error_constraints.add(
                    expr=model.Lt[t] >= model.Nt[t] - model.Nkt[k, t] - (1 - model.c[k, t]) * n
                )
                model.leaf_error_constraints.add(
                    expr=model.Lt[t] <= model.Nt[t] - model.Nkt[k, t] + model.c[k, t] * n
                )

        model.parent_branching_constraints = ConstraintList()
        for i in n_range:
            for t in leaf_ndoes:
                left_ancestors, right_ancestors = self.__ancestors(t)
                for m in right_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) >= model.bt[m] - (1 - model.z[i, t]) * self.M
                    )
                for m in left_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) + self.epsilon <= model.bt[m] + (1 - model.z[i, t]) * (self.M + self.epsilon)
                    )
        for t in parent_nodes:
            model.parent_branching_constraints.add(
                expr=model.bt[t] <= model.d[t]
            )

        # Objective
        model.obj = Objective(
            expr=sum([model.Lt[t] for t in leaf_ndoes]) / L_hat + sum([model.d[t] for t in parent_nodes]) * self.alpha
        )

        return model

    def __parent(self, i: int):
        assert i > 1, "Root node (i=1) doesn't have parent! "
        assert i <= 2**(self.D + 1), "Out of nodes index! Total: {0}; i: {1}".format(2**(self.D + 1), i)
        return int(i/2)

    def __ancestors(self, i: int):
        assert i > 1, "Root node (i=1) doesn't have ancestors! "
        assert i <= 2**(self.D + 1), "Out of nodes index! Total: {0}; i: {1}".format(2**(self.D + 1), i)
        left_ancestors = []
        right_ancestors = []
        j = i
        while j > 1:
            if j % 2 == 0:
                left_ancestors.append(int(j/2))
            else:
                right_ancestors.append(int(j/2))
            j = int(j/2)
        return left_ancestors, right_ancestors


class OptimalHyperTreeModel:
    def __init__(self, x_cols: list, y_col: str, tree_depth: int, N_min: int, alpha: float = 0.01,
                 M: int = 1e6, epsilon: float = 1e-4,
                 solver_name: str = "cplex"):
        self.y_col = y_col
        self.P = len(x_cols)
        self.P_range = x_cols
        self.K_range = None
        self.solver_name = solver_name
        self.D = tree_depth
        self.Nmin = N_min
        self.M = M
        self.epsilon = epsilon
        self.alpha = alpha
        self.is_trained = False
        nodes = list(range(1, (2 ** (self.D + 1))))
        self.parent_nodes = nodes[0: 2 ** (self.D + 1) - 2 ** self.D - 1]
        self.leaf_ndoes = nodes[-2 ** self.D:]
        self.normalizer = {}

        # solutions
        self.l = None
        self.c = None
        self.d = None
        self.a = None
        self.b = None
        self.Nt = None
        self.Nkt = None
        self.Lt = None

        assert tree_depth > 0, "Tree depth must be greater than 0! (Actual: {0})".format(tree_depth)

    def train(self, data: pd.DataFrame, show_training_process: bool = True):
        data = data.copy()
        for col in self.P_range:
            col_max = max(data[col])
            col_min = min(data[col])
            self.normalizer[col] = (col_max, col_min)
            if col_max != col_min:
                data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                data[col] = 1
        model = self.__generate_model(data.reset_index(drop=True))
        solver = SolverFactory(self.solver_name)
        res = solver.solve(model, tee=show_training_process)
        status = str(res.solver.termination_condition)
        self.is_trained = True
        loss = value(model.obj)
        self.l = {t: value(model.l[t]) for t in self.leaf_ndoes}
        self.c = {t: [value(model.c[k, t]) for k in self.K_range] for t in self.leaf_ndoes}
        self.d = {t: value(model.d[t]) for t in self.parent_nodes}
        self.a = {t: [value(model.ajt[j, t]) for j in self.P_range] for t in self.parent_nodes}
        self.b = {t: value(model.bt[t]) for t in self.parent_nodes}
        self.Nt = {t: value(model.Nt[t]) for t in self.leaf_ndoes}
        self.Nkt = {t: [value(model.Nkt[k, t]) for k in self.K_range] for t in self.leaf_ndoes}
        self.Lt = {t: value(model.Lt[t]) for t in self.leaf_ndoes}
        logging.info("Training done. Loss: {1}. Optimization status: {0}".format(status, loss))

    def predict(self, data: pd.DataFrame):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet! Please use `train()` to train the model first!")

        new_data = data.copy()
        new_data_cols = data.columns
        for col in self.P_range:
            if col not in new_data_cols:
                raise ValueError("Column {0} is not in the given data for prediction! ".format(col))
            col_max, col_min = self.normalizer[col]
            if col_max != col_min:
                new_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                new_data[col] = 1

        prediction = []
        for j in range(new_data.shape[0]):
            x = np.array([new_data.ix[j, i] for i in self.P_range])
            t = 1
            d = 0
            while d < self.D:
                at = np.array(self.a[t])
                bt = self.b[t]
                if at.dot(x) < bt:
                    t = t * 2
                else:
                    t = t * 2 + 1
                d = d + 1
            y_hat = self.c[t]
            prediction.append(self.K_range[y_hat.index(max(y_hat))])
        data["prediction"] = prediction
        return data

    def __generate_model(self, data: pd.DataFrame):
        model = ConcreteModel(name="OptimalTreeModel")
        n = data.shape[0]
        label = data[[self.y_col]]
        label["__value__"] = 1
        Y = label.pivot(columns=self.y_col, values="__value__")

        L_hat = max(label.groupby(by=self.y_col).sum()["__value__"]) / n

        Y.fillna(value=-1, inplace=True)

        n_range = range(n)
        K_range = sorted(list(set(data[self.y_col])))
        P_range = self.P_range

        self.K_range = K_range

        parent_nodes = self.parent_nodes
        leaf_ndoes = self.leaf_ndoes

        # Variables
        model.z = Var(n_range, leaf_ndoes, within=Binary)
        model.l = Var(leaf_ndoes, within=Binary)
        model.c = Var(K_range, leaf_ndoes, within=Binary)
        model.d = Var(parent_nodes, within=Binary)
        model.s = Var(P_range, parent_nodes, within=Binary)

        model.Nt = Var(leaf_ndoes, within=NonNegativeReals)
        model.Nkt = Var(K_range, leaf_ndoes, within=NonNegativeReals)
        model.Lt = Var(leaf_ndoes, within=NonNegativeReals)
        model.ajt = Var(P_range, parent_nodes)
        model.bt = Var(parent_nodes)
        model.a_hat_jt = Var(P_range, parent_nodes, within=NonNegativeReals)

        # Constraints
        model.integer_relationship_constraints = ConstraintList()
        for t in leaf_ndoes:
            model.integer_relationship_constraints.add(
                expr=sum([model.c[k, t] for k in K_range]) == model.l[t]
            )
        for i in n_range:
            for t in leaf_ndoes:
                model.integer_relationship_constraints.add(
                    expr=model.z[i, t] <= model.l[t]
                )
        for i in n_range:
            model.integer_relationship_constraints.add(
                expr=sum([model.z[i, t] for t in leaf_ndoes]) == 1
            )
        for t in leaf_ndoes:
            model.integer_relationship_constraints.add(
                expr=sum([model.z[i, t] for i in n_range]) >= model.l[t] * self.Nmin
            )
        for j in P_range:
            for t in parent_nodes:
                model.integer_relationship_constraints.add(
                    expr=model.s[j, t] <= model.d[t]
                )
        for t in parent_nodes:
            model.integer_relationship_constraints.add(
                expr=sum([model.s[j, t] for j in P_range]) >= model.d[t]
            )
        for t in parent_nodes:
            if t != 1:
                model.integer_relationship_constraints.add(
                    expr=model.d[t] <= model.d[self.__parent(t)]
                )

        model.leaf_samples_constraints = ConstraintList()
        for t in leaf_ndoes:
            model.leaf_samples_constraints.add(
                expr=model.Nt[t] == sum([model.z[i, t] for i in n_range])
            )
        for t in leaf_ndoes:
            for k in K_range:
                model.leaf_samples_constraints.add(
                    expr=model.Nkt[k, t] == sum([model.z[i, t] * (1 + Y.loc[i, k]) / 2.0 for i in n_range])
                )

        model.leaf_error_constraints = ConstraintList()
        for k in K_range:
            for t in leaf_ndoes:
                model.leaf_error_constraints.add(
                    expr=model.Lt[t] >= model.Nt[t] - model.Nkt[k, t] - (1 - model.c[k, t]) * n
                )
                model.leaf_error_constraints.add(
                    expr=model.Lt[t] <= model.Nt[t] - model.Nkt[k, t] + model.c[k, t] * n
                )

        model.parent_branching_constraints = ConstraintList()
        for i in n_range:
            for t in leaf_ndoes:
                left_ancestors, right_ancestors = self.__ancestors(t)
                for m in right_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.ajt[j, m] * data.loc[i, j] for j in P_range]) >= model.bt[m] - (1 - model.z[i, t]) * self.M
                    )
                for m in left_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.ajt[j, m] * data.loc[i, j] for j in P_range]) + self.epsilon <= model.bt[m] + (1 - model.z[i, t]) * (self.M + self.epsilon)
                    )
        for t in parent_nodes:
            model.parent_branching_constraints.add(
                expr=sum([model.a_hat_jt[j, t] for j in P_range]) <= model.d[t]
            )
        for j in P_range:
            for t in parent_nodes:
                model.parent_branching_constraints.add(
                    expr=model.a_hat_jt[j, t] >= model.ajt[j, t]
                )
                model.parent_branching_constraints.add(
                    expr=model.a_hat_jt[j, t] >= -model.ajt[j, t]
                )
                model.parent_branching_constraints.add(
                    expr=model.ajt[j, t] >= -model.s[j, t]
                )
                model.parent_branching_constraints.add(
                    expr=model.ajt[j, t] <= model.s[j, t]
                )
        for t in parent_nodes:
            model.parent_branching_constraints.add(
                expr=model.bt[t] >= -model.d[t]
            )
            model.parent_branching_constraints.add(
                expr=model.bt[t] <= model.d[t]
            )

        # Objective
        model.obj = Objective(
            expr=sum([model.Lt[t] for t in leaf_ndoes]) / L_hat + sum([model.s[j, t] for j in P_range for t in parent_nodes]) * self.alpha
        )

        return model

    def __parent(self, i: int):
        assert i > 1, "Root node (i=1) doesn't have parent! "
        assert i <= 2**(self.D + 1), "Out of nodes index! Total: {0}; i: {1}".format(2**(self.D + 1), i)
        return int(i/2)

    def __ancestors(self, i: int):
        assert i > 1, "Root node (i=1) doesn't have ancestors! "
        assert i <= 2**(self.D + 1), "Out of nodes index! Total: {0}; i: {1}".format(2**(self.D + 1), i)
        left_ancestors = []
        right_ancestors = []
        j = i
        while j > 1:
            if j % 2 == 0:
                left_ancestors.append(int(j/2))
            else:
                right_ancestors.append(int(j/2))
            j = int(j/2)
        return left_ancestors, right_ancestors


if __name__ == "__main__":
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
    model = OptimalHyperTreeModel(["x1", "x2"], "y", tree_depth=2, N_min=1, alpha=0.1)
    model.train(data)

    print(model.predict(test_data))


