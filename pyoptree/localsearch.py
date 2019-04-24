import numpy as np
from abc import abstractmethod, ABCMeta
from pyoptree.tree import Tree, TreeModel
import logging


class AbstractOptimalTreeModelOptimizer(metaclass=ABCMeta):
    def __init__(self, Nmin: int):
        self.Nmin = Nmin

    @staticmethod
    def shuffle(index_set: list):
        index_set_bk = index_set.copy()
        np.random.shuffle(index_set_bk)
        return index_set_bk

    def local_search(self, tree: Tree, x, y):
        tree = tree.copy()
        error_previous = tree.loss(x, y)
        error_current = np.inf

        logging.info("Current error of the whole tree: {0}".format(error_previous))
        i = 1
        while True:
            for t in AbstractOptimalTreeModelOptimizer.shuffle(tree.get_parent_nodes()):
                logging.info("Visiting node {0}...".format(t))
                subtree = tree.subtree(t)
                res = tree.evaluate(x)
                L = []
                for tt in subtree.get_leaf_nodes():
                    for ii in res[tt]:
                        L.append(ii)
                if len(L) > 0:
                    logging.info("Training in {0}th iteration...".format(i))
                    i += 1
                    new_subtree = self.optimize_node_parallel(subtree, x, y, L)
                    tree.a[t] = new_subtree.a[t]
                    tree.b[t] = new_subtree.b[t]

                    error_current = tree.loss(x, y)

                    logging.info("Current error of the whole tree: {0}".format(error_current))

            if round(error_current, 5) == round(error_previous, 5):
                tree.generate_majority_leaf_class(x, y)
                return tree

            error_previous = error_current

    def optimize_node_parallel(self, subtree: Tree, x, y, L):
        new_sub_tree = subtree.copy()
        sub_x = x[L, ::]
        sub_y = y[L]

        lower_tree, upper_tree = subtree.children()
        error_best = new_sub_tree.loss(sub_x, sub_y)

        logging.debug("Current best error of the subtree: {0}".format(error_best))

        updated = False
        para_tree, error_para = self.best_split(lower_tree, upper_tree, sub_x,
                                                sub_y)
        if error_para < error_best:
            logging.info("Updating by parallel split")
            new_sub_tree.a[new_sub_tree.root_node] = para_tree.a[para_tree.root_node]
            new_sub_tree.b[new_sub_tree.root_node] = para_tree.b[para_tree.root_node]
            error_best = error_para
            updated = True

        error_lower = lower_tree.loss(sub_x, sub_y)
        if error_lower < error_best and lower_tree.depth > 0:
            logging.info("Updating by replacing by lower child tree")
            new_sub_tree.a[new_sub_tree.root_node] = lower_tree.a[lower_tree.root_node]
            new_sub_tree.b[new_sub_tree.root_node] = lower_tree.b[lower_tree.root_node]
            error_best = error_lower
            updated = True

        error_upper = upper_tree.loss(sub_x, sub_y)
        if error_upper < error_best and upper_tree.depth > 0:
            logging.info("Updating by replacing by upper child tree")
            new_sub_tree.a[new_sub_tree.root_node] = upper_tree.a[upper_tree.root_node]
            new_sub_tree.b[new_sub_tree.root_node] = upper_tree.b[upper_tree.root_node]
            error_best = error_upper
            updated = True

        if not updated:
            logging.info("No update, return the original tree")

        return new_sub_tree

    @abstractmethod
    def best_split(self, lower_tree: Tree, upper_tree: Tree, x, y):
        pass


class OptimalTreeModelOptimizer(AbstractOptimalTreeModelOptimizer):
    def best_split(self, lower_tree: Tree, upper_tree: Tree, x, y):
        assert lower_tree.root_node % 2 == 0, "Illegal lower tree! (root node: {0})".format(lower_tree.root_node)
        assert upper_tree.root_node == lower_tree.root_node + 1, "Illegal upper tree! (lower tree root node: {0}, " \
                                                                 "upper tree root node: {1})".format(
            lower_tree.root_node,
            upper_tree.root_node)
        assert lower_tree.depth == upper_tree.depth, "Unequal depth of lower tree and upper tree! ({0} != {1})".format(
            lower_tree.depth, upper_tree.depth
        )

        n, p = x.shape
        error_best = np.inf
        best_tree = lower_tree

        logging.debug("Calculating best parallel split for {0} points with dimension {1}".format(n, p))
        for j in range(p):
            logging.debug("Visiting {0}th dimension. Current best error of the subtree: {1}".format(j, error_best))
            values = x[::, j]
            values = sorted(values)
            for i in range(n - 1):
                b = (values[i] + values[i + 1]) / 2

                parent_node = int(lower_tree.root_node / 2)
                parent_node_a = np.zeros(p)
                parent_node_a[j] = 1
                parent_a = {**{parent_node: parent_node_a}, **lower_tree.a, **upper_tree.a}
                parent_b = {**{parent_node: b}, **lower_tree.b, **upper_tree.b}
                parent_tree = Tree(parent_node, lower_tree.depth + 1, parent_a, parent_b)

                min_leaf_size = parent_tree.min_leaf_size(x)

                if min_leaf_size >= self.Nmin:
                    error = parent_tree.loss(x, y)

                    if error < error_best:
                        error_best = error

                        best_tree = parent_tree

        logging.debug("Complete calculating best parallel split")

        return best_tree, error_best
