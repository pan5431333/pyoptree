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
from abc import abstractmethod, ABCMeta
from sklearn.tree import DecisionTreeClassifier
from inspect import getmembers

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', )


class AbstractOptimalTreeModel(metaclass=ABCMeta):
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
        self.parent_nodes, self.leaf_ndoes = self.generate_nodes(self.D)
        self.normalizer = {}

        # optimization model
        self.model = None

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

    def train(self, data: pd.DataFrame, show_training_process: bool = True, warm_start: bool = True):
        data = data.copy().reset_index(drop=True)
        for col in self.P_range:
            col_max = max(data[col])
            col_min = min(data[col])
            self.normalizer[col] = (col_max, col_min)
            if col_max != col_min:
                data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                data[col] = 1

        solver = SolverFactory(self.solver_name)
        solver.options["LPMethod"] = 4

        start_tree_depth = 1 if warm_start else self.D

        global_status = "Not started"
        global_loss = np.inf
        previous_depth_params = None
        for d in range(start_tree_depth, self.D + 1):
            if d < self.D:
                logging.info("Warm starting the optimization with tree depth {0} / {1}...".format(d, self.D))
            else:
                logging.info("Optimizing the tree with depth {0}...".format(self.D))

            warm_start_params = None
            if warm_start:
                cart_params = self._get_cart_params(data, d)
                warm_start_params = self._select_better_warm_start_params([previous_depth_params, cart_params], data)

            parent_nodes, leaf_nodes = self.generate_nodes(d)
            model = self.generate_model(data, parent_nodes, leaf_nodes, warm_start_params)

            res = solver.solve(model, tee=show_training_process, warmstart=True)
            status = str(res.solver.termination_condition)
            loss = value(model.obj)

            previous_depth_params = self._generate_warm_start_params_from_previous_depth(model, data.shape[0],
                                                                                         parent_nodes, leaf_nodes)

            if d == self.D:
                global_status = status
                global_loss = loss
                self.model = model
                self.is_trained = True
                self.l = {t: value(model.l[t]) for t in self.leaf_ndoes}
                self.c = {t: [value(model.c[k, t]) for k in self.K_range] for t in self.leaf_ndoes}
                self.d = {t: value(model.d[t]) for t in self.parent_nodes}
                self.a = {t: [value(model.a[j, t]) for j in self.P_range] for t in self.parent_nodes}
                self.b = {t: value(model.bt[t]) for t in self.parent_nodes}
                self.Nt = {t: value(model.Nt[t]) for t in self.leaf_ndoes}
                self.Nkt = {t: [value(model.Nkt[k, t]) for k in self.K_range] for t in self.leaf_ndoes}
                self.Lt = {t: value(model.Lt[t]) for t in self.leaf_ndoes}

        logging.info("Training done. Loss: {1}. Optimization status: {0}".format(global_status, global_loss))
        logging.info("Training done(Contd.): training accuracy: {0}".format(1 - sum(self.Lt.values()) / data.shape[0]))

    @abstractmethod
    def generate_model(self, data: pd.DataFrame, parent_nodes: list, leaf_nodes: list, warm_start_params: dict = None):
        """Generate the corresponding model instance"""
        pass

    def _get_cart_params(self, data: pd.DataFrame, depth: int):
        cart_model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=self.Nmin)
        clf = cart_model.fit(data[self.P_range].values.tolist(), data[[self.y_col]].values.tolist())
        members = getmembers(clf.tree_)
        members_dict = {m[0]: m[1] for m in members}
        self.K_range = sorted(list(set(data[self.y_col])))
        cart_params = self._convert_skcart_to_params(members_dict)

        parent_nodes, leaf_nodes = self.generate_nodes(depth)
        z = {(i, t): 0 for i in range(0, data.shape[0]) for t in leaf_nodes}
        for i in range(data.shape[0]):
            xi = np.array([data.ix[i, j] for j in self.P_range])
            node = 1
            current_depth = 0
            while current_depth < depth:
                current_b = xi.dot(np.array([cart_params["a"][j, node] for j in self.P_range]))
                if current_b < cart_params["bt"][node]:
                    node = 2 * node
                else:
                    node = 2 * node + 1
                current_depth += 1
                z[i, node] = 1
        cart_params["z"] = z

        return cart_params

    @abstractmethod
    def _convert_skcart_to_params(self, clf):
        pass

    def _select_better_warm_start_params(self, params_list: list, data: pd.DataFrame):
        params_list = [p for p in params_list if p is not None]
        if len(params_list) == 0:
            return None

        n = data.shape[0]
        label = data[[self.y_col]].copy()
        label["__value__"] = 1
        L_hat = max(label.groupby(by=self.y_col).sum()["__value__"]) / n

        best_params = None
        current_loss = np.inf
        for i, params in enumerate(params_list):
            loss = self._get_solution_loss(params, L_hat)
            logging.info("Loss of the {0}th warmstart solution is: {1}. The current best loss is: {2}.".format(i, loss,
                                                                                                               current_loss))
            if loss < current_loss:
                current_loss = loss
                best_params = params

        return best_params

    @abstractmethod
    def _get_solution_loss(self, params, L_hat: float):
        pass

    @abstractmethod
    def _generate_warm_start_params_from_previous_depth(self, model, n_training_data: int,
                                                        parent_nodes: list, leaf_nodes: list):
        pass

    def get_feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet! Please use `train()` to train the model first!")

        return self._feature_importance()

    @abstractmethod
    def _feature_importance(self):
        pass

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

    def _parent(self, i: int):
        assert i > 1, "Root node (i=1) doesn't have parent! "
        assert i <= 2 ** (self.D + 1), "Out of nodes index! Total: {0}; i: {1}".format(2 ** (self.D + 1), i)
        return int(i / 2)

    def _ancestors(self, i: int):
        assert i > 1, "Root node (i=1) doesn't have ancestors! "
        assert i <= 2 ** (self.D + 1), "Out of nodes index! Total: {0}; i: {1}".format(2 ** (self.D + 1), i)
        left_ancestors = []
        right_ancestors = []
        j = i
        while j > 1:
            if j % 2 == 0:
                left_ancestors.append(int(j / 2))
            else:
                right_ancestors.append(int(j / 2))
            j = int(j / 2)
        return left_ancestors, right_ancestors

    @staticmethod
    def generate_nodes(tree_depth: int):
        nodes = list(range(1, int(2 ** (tree_depth + 1))))
        parent_nodes = nodes[0: 2 ** (tree_depth + 1) - 2 ** tree_depth - 1]
        leaf_ndoes = nodes[-2 ** tree_depth:]
        return parent_nodes, leaf_ndoes

    @staticmethod
    def positive_or_zero(i: float):
        if i >= 0:
            return i
        else:
            return 0

    @staticmethod
    def convert_to_complete_tree(incomplete_tree: dict):
        children_left = incomplete_tree["children_left"]
        children_right = incomplete_tree["children_right"]
        depth = incomplete_tree["max_depth"]

        mapping = {1: 0}
        for t in range(2, 2 ** (depth + 1)):
            parent_of_t = int(t / 2)
            parent_in_original_tree = mapping[parent_of_t]
            is_left_child = t % 2 == 0

            if is_left_child:
                node_in_original_tree = children_left[parent_in_original_tree]
            else:
                node_in_original_tree = children_right[parent_in_original_tree]
            mapping[t] = node_in_original_tree

        return mapping

    @staticmethod
    def get_leaf_mapping(tree_nodes_mapping: dict):
        number_nodes = len(tree_nodes_mapping)
        depth = int(np.log2(number_nodes + 1) - 1)
        nodes = list(range(1, number_nodes + 1))
        leaf_nodes = nodes[-2 ** depth:]
        leaf_nodes_mapping = {}
        for t in leaf_nodes:
            tt = t
            while tt >= 1:
                original_t = tree_nodes_mapping[tt]
                if original_t != -1:
                    leaf_nodes_mapping[t] = original_t
                    break
                else:
                    tt = int(tt / 2)
        return leaf_nodes_mapping


class OptimalTreeModel(AbstractOptimalTreeModel):
    def generate_model(self, data: pd.DataFrame, parent_nodes: list, leaf_ndoes: list, warm_start_params: dict = None):
        model = ConcreteModel(name="OptimalTreeModel")
        n = data.shape[0]
        label = data[[self.y_col]].copy()
        label["__value__"] = 1
        Y = label.pivot(columns=self.y_col, values="__value__")

        L_hat = max(label.groupby(by=self.y_col).sum()["__value__"]) / n

        Y.fillna(value=-1, inplace=True)

        n_range = range(n)
        K_range = sorted(list(set(data[self.y_col])))
        P_range = self.P_range

        self.K_range = K_range

        warm_start_params = {} if warm_start_params is None else warm_start_params

        # Variables
        model.z = Var(n_range, leaf_ndoes, within=Binary, initialize=warm_start_params.get("z"))
        model.l = Var(leaf_ndoes, within=Binary, initialize=warm_start_params.get("l"))
        model.c = Var(K_range, leaf_ndoes, within=Binary, initialize=warm_start_params.get("c"))
        model.d = Var(parent_nodes, within=Binary, initialize=warm_start_params.get("d"))
        model.a = Var(P_range, parent_nodes, within=Binary, initialize=warm_start_params.get("a"))

        model.Nt = Var(leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Nt"))
        model.Nkt = Var(K_range, leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Nkt"))
        model.Lt = Var(leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Lt"))
        model.bt = Var(parent_nodes, within=NonNegativeReals, initialize=warm_start_params.get("bt"))

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
                    expr=model.d[t] <= model.d[self._parent(t)]
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
                left_ancestors, right_ancestors = self._ancestors(t)
                for m in right_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) >= model.bt[m] - (1 - model.z[
                            i, t]) * self.M
                    )
                for m in left_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) + self.epsilon <= model.bt[m] + (1 -
                                                                                                                     model.z[
                                                                                                                         i, t]) * (
                                                                                                                        self.M + self.epsilon)
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

    def _feature_importance(self):
        importance_scores = np.array([self.a[t] for t in self.a]).sum(axis=0)
        return {x: s for x, s in zip(self.P_range, importance_scores)}

    def _generate_warm_start_params_from_previous_depth(self, model, n_training_data: int,
                                                        parent_nodes: list, leaf_nodes: list):
        ret = {}
        D = int(np.log2(len(leaf_nodes))) + 1
        new_parent_nodes, new_leaf_nodes = self.generate_nodes(D)
        n_range = range(n_training_data)

        ret["z"] = {(i, t): int(value(model.z[i, int(t / 2)])) if t % 2 == 1 else 0 for i in n_range for t in
                    new_leaf_nodes}
        ret["l"] = {t: int(value(model.l[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret["c"] = {(k, t): int(value(model.c[k, int(t / 2)])) if t % 2 == 1 else 0 for k in self.K_range for t in
                    new_leaf_nodes}
        ret_d_1 = {t: int(value(model.d[t])) for t in parent_nodes}
        ret_d_2 = {t: 0 for t in leaf_nodes}
        ret["d"] = {**ret_d_1, **ret_d_2}
        ret_a_1 = {(j, t): int(value(model.a[j, t])) for j in self.P_range for t in parent_nodes}
        ret_a_2 = {(j, t): 0 for j in self.P_range for t in leaf_nodes}
        ret["a"] = {**ret_a_1, **ret_a_2}
        ret["Nt"] = {t: self.positive_or_zero(value(model.Nt[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret["Nkt"] = {(k, t): self.positive_or_zero(value(model.Nkt[k, int(t / 2)])) if t % 2 == 1 else 0 for k in
                      self.K_range for t in new_leaf_nodes}
        ret["Lt"] = {t: self.positive_or_zero(value(model.Lt[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret_b_1 = {t: self.positive_or_zero(value(model.bt[t])) for t in parent_nodes}
        ret_b_2 = {t: 0 for t in leaf_nodes}
        ret["bt"] = {**ret_b_1, **ret_b_2}
        return ret

    def _convert_skcart_to_params(self, members: dict):
        complete_incomplete_nodes_mapping = self.convert_to_complete_tree(members)
        leaf_nodes_mapping = self.get_leaf_mapping(complete_incomplete_nodes_mapping)
        D = members["max_depth"]
        ret = {}
        parent_nodes, leaf_nodes = self.generate_nodes(D)

        ret["l"] = {t: self.extract_solution_l(complete_incomplete_nodes_mapping, t) for t in leaf_nodes}
        ret_c_helper = {t: [1 if s else 0 for s in np.array(members["value"][leaf_nodes_mapping[t]][0]) == max(
            members["value"][leaf_nodes_mapping[t]][0])] for t in leaf_nodes}
        ret["c"] = {(k, t): ret_c_helper[t][kk] for kk, k in enumerate(self.K_range) for t in leaf_nodes}
        ret["d"] = {t: 1 if (complete_incomplete_nodes_mapping[t] != -1 and
                             members["children_left"][complete_incomplete_nodes_mapping[t]] != -1 and
                             members["children_right"][complete_incomplete_nodes_mapping[t]] != -1) else 0
                    for t in parent_nodes}
        ret["a"] = {(j, t): self.extract_solution_a(members, complete_incomplete_nodes_mapping, j, t) for j in
                    self.P_range
                    for t in parent_nodes}
        ret["Nt"] = {t: members["n_node_samples"][leaf_nodes_mapping[t]] for t in leaf_nodes}
        ret["Nkt"] = {(k, t): members["value"][leaf_nodes_mapping[t]][0][kk] for kk, k in enumerate(self.K_range) for t
                      in leaf_nodes}
        ret["Lt"] = {t: sum(
            [i for i in members["value"][leaf_nodes_mapping[t]][0] if
             i != max(members["value"][leaf_nodes_mapping[t]][0])])
            for t in leaf_nodes}
        ret["bt"] = {t: 0 if members["threshold"][complete_incomplete_nodes_mapping[t]] <= 0 else
        members["threshold"][complete_incomplete_nodes_mapping[t]] for t in parent_nodes}

        return ret

    def _get_solution_loss(self, params: dict, L_hat: float):
        return sum(params["Lt"].values()) / L_hat + self.alpha * sum(params["d"].values())

    def extract_solution_a(self, members: dict, nodes_mapping: dict, j: str, t: int):
        if nodes_mapping[t] == -1:
            return 0

        feature = members["feature"][nodes_mapping[t]]
        if feature < 0:
            return 0

        if self.P_range[feature] == j:
            return 1
        else:
            return 0

    @staticmethod
    def extract_solution_l(nodes_mapping: dict, t: int):
        if nodes_mapping[t] > -1:
            return 1

        if t % 2 == 0:
            return 0
        else:
            return 1


class OptimalHyperTreeModel(AbstractOptimalTreeModel):
    def generate_model(self, data: pd.DataFrame, parent_nodes: list, leaf_ndoes: list, warm_start_params: dict = None):
        model = ConcreteModel(name="OptimalTreeModel")
        n = data.shape[0]
        label = data[[self.y_col]].copy()
        label["__value__"] = 1
        Y = label.pivot(columns=self.y_col, values="__value__")

        L_hat = max(label.groupby(by=self.y_col).sum()["__value__"]) / n

        Y.fillna(value=-1, inplace=True)

        n_range = range(n)
        K_range = sorted(list(set(data[self.y_col])))
        P_range = self.P_range

        self.K_range = K_range

        warm_start_params = {} if warm_start_params is None else warm_start_params

        # Variables
        model.z = Var(n_range, leaf_ndoes, within=Binary, initialize=warm_start_params.get("z"))
        model.l = Var(leaf_ndoes, within=Binary, initialize=warm_start_params.get("l"))
        model.c = Var(K_range, leaf_ndoes, within=Binary, initialize=warm_start_params.get("c"))
        model.d = Var(parent_nodes, within=Binary, initialize=warm_start_params.get("d"))
        model.s = Var(P_range, parent_nodes, within=Binary, initialize=warm_start_params.get("s"))

        model.Nt = Var(leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Nt"))
        model.Nkt = Var(K_range, leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Nkt"))
        model.Lt = Var(leaf_ndoes, within=NonNegativeReals, initialize=warm_start_params.get("Lt"))
        model.a = Var(P_range, parent_nodes, initialize=warm_start_params.get("a"))
        model.bt = Var(parent_nodes, initialize=warm_start_params.get("bt"))
        model.a_hat_jt = Var(P_range, parent_nodes, within=NonNegativeReals,
                             initialize=warm_start_params.get("a_hat_jt"))

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
                    expr=model.d[t] <= model.d[self._parent(t)]
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
                left_ancestors, right_ancestors = self._ancestors(t)
                for m in right_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) >= model.bt[m] - (1 - model.z[
                            i, t]) * self.M
                    )
                for m in left_ancestors:
                    model.parent_branching_constraints.add(
                        expr=sum([model.a[j, m] * data.loc[i, j] for j in P_range]) + self.epsilon <= model.bt[m] + (
                                                                                                                        1 -
                                                                                                                        model.z[
                                                                                                                            i, t]) * (
                                                                                                                        self.M + self.epsilon)
                    )
        for t in parent_nodes:
            model.parent_branching_constraints.add(
                expr=sum([model.a_hat_jt[j, t] for j in P_range]) <= model.d[t]
            )
        for j in P_range:
            for t in parent_nodes:
                model.parent_branching_constraints.add(
                    expr=model.a_hat_jt[j, t] >= model.a[j, t]
                )
                model.parent_branching_constraints.add(
                    expr=model.a_hat_jt[j, t] >= -model.a[j, t]
                )
                model.parent_branching_constraints.add(
                    expr=model.a[j, t] >= -model.s[j, t]
                )
                model.parent_branching_constraints.add(
                    expr=model.a[j, t] <= model.s[j, t]
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
            expr=sum([model.Lt[t] for t in leaf_ndoes]) / L_hat + sum(
                [model.s[j, t] for j in P_range for t in parent_nodes]) * self.alpha
        )

        return model

    def _feature_importance(self):
        importance_scores = np.array(
            [[value(self.model.s[j, t]) * value(self.model.a_hat_jt[j, t]) for j in self.P_range] for t in
             self.parent_nodes]).sum(axis=0)
        return {x: s for x, s in zip(self.P_range, importance_scores)}

    def _generate_warm_start_params_from_previous_depth(self, model, n_training_data: int,
                                                        parent_nodes: list, leaf_nodes: list):
        ret = {}
        D = int(np.log2(len(leaf_nodes))) + 1
        new_parent_nodes, new_leaf_nodes = self.generate_nodes(D)
        n_range = range(n_training_data)

        ret["z"] = {(i, t): int(value(model.z[i, int(t / 2)])) if t % 2 == 1 else 0 for i in n_range for t in
                    new_leaf_nodes}
        ret["l"] = {t: int(value(model.l[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret["c"] = {(k, t): int(value(model.c[k, int(t / 2)])) if t % 2 == 1 else 0 for k in self.K_range for t in
                    new_leaf_nodes}
        ret_d_1 = {t: int(value(model.d[t])) for t in parent_nodes}
        ret_d_2 = {t: 0 for t in leaf_nodes}
        ret["d"] = {**ret_d_1, **ret_d_2}
        ret_s_1 = {(j, t): int(value(model.s[j, t])) for j in self.P_range for t in parent_nodes}
        ret_s_2 = {(j, t): 0 for j in self.P_range for t in leaf_nodes}
        ret["s"] = {**ret_s_1, **ret_s_2}
        ret["Nt"] = {t: self.positive_or_zero(value(model.Nt[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret["Nkt"] = {(k, t): self.positive_or_zero(value(model.Nkt[k, int(t / 2)])) if t % 2 == 1 else 0 for k in
                      self.K_range for t in new_leaf_nodes}
        ret["Lt"] = {t: self.positive_or_zero(value(model.Lt[int(t / 2)])) if t % 2 == 1 else 0 for t in new_leaf_nodes}
        ret_a_1 = {(j, t): value(model.a[j, t]) for j in self.P_range for t in parent_nodes}
        ret_a_2 = {(j, t): 0 for j in self.P_range for t in leaf_nodes}
        ret["a"] = {**ret_a_1, **ret_a_2}
        ret_b_1 = {t: value(model.bt[t]) for t in parent_nodes}
        ret_b_2 = {t: 0 for t in leaf_nodes}
        ret["bt"] = {**ret_b_1, **ret_b_2}
        ret_a_hat_jt_1 = {(j, t): self.positive_or_zero(value(model.a_hat_jt[j, t])) for j in self.P_range for t in
                          parent_nodes}
        ret_a_hat_jt_2 = {(j, t): 0 for j in self.P_range for t in parent_nodes}
        ret["a_hat_jt"] = {**ret_a_hat_jt_1, **ret_a_hat_jt_2}
        return ret

    def _convert_skcart_to_params(self, members: dict):
        complete_incomplete_nodes_mapping = self.convert_to_complete_tree(members)
        leaf_nodes_mapping = self.get_leaf_mapping(complete_incomplete_nodes_mapping)
        D = members["max_depth"]
        ret = {}
        parent_nodes, leaf_nodes = self.generate_nodes(D)

        ret["l"] = {t: OptimalTreeModel.extract_solution_l(complete_incomplete_nodes_mapping, t) for t in leaf_nodes}
        ret_c_helper = {t: [1 if s else 0 for s in np.array(members["value"][leaf_nodes_mapping[t]][0]) == max(
            members["value"][leaf_nodes_mapping[t]][0])] for t in leaf_nodes}
        ret["c"] = {(k, t): ret_c_helper[t][kk] for kk, k in enumerate(self.K_range) for t in leaf_nodes}
        ret["d"] = {t: 1 if (complete_incomplete_nodes_mapping[t] != -1 and
                             members["children_left"][complete_incomplete_nodes_mapping[t]] != -1) else 0
                    for t in parent_nodes}
        ret["s"] = {(j, t): self.extract_solution_s(members, complete_incomplete_nodes_mapping, j, t) for j in
                    self.P_range
                    for t in parent_nodes}
        ret["Nt"] = {t: members["n_node_samples"][leaf_nodes_mapping[t]] for t in leaf_nodes}
        ret["Nkt"] = {(k, t): members["value"][leaf_nodes_mapping[t]][0][kk] for kk, k in enumerate(self.K_range) for t
                      in leaf_nodes}
        ret["Lt"] = {t: sum(
            [i for i in members["value"][leaf_nodes_mapping[t]][0] if
             i != max(members["value"][leaf_nodes_mapping[t]][0])])
            for t in leaf_nodes}
        ret["a"] = ret["s"]
        ret["a_hat_jt"] = ret["s"]
        ret["bt"] = {t: 0 if members["threshold"][complete_incomplete_nodes_mapping[t]] <= 0 else
        members["threshold"][complete_incomplete_nodes_mapping[t]] for t in parent_nodes}

        return ret

    def _get_solution_loss(self, params: dict, L_hat: float):
        return sum(params["Lt"].values()) / L_hat + self.alpha * sum(params["s"].values())

    def extract_solution_s(self, members: dict, nodes_mapping: dict, j: str, t: int):
        if nodes_mapping[t] == -1:
            return 0

        feature = members["feature"][nodes_mapping[t]]
        if feature < 0:
            return 0

        if self.P_range[feature] == j:
            return 1
        else:
            return 0


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
