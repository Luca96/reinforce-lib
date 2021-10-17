"""Decision trees (DTs)"""

import numpy as np

from rl import utils
from typing import List


class TreeNode:
    INF = np.reshape(np.inf, newshape=[-1])

    # TODO: memory leak; store data only in "root", then only "indices" for the other nodes.
    def __init__(self, x: np.ndarray, y: np.ndarray, parent=None, min_samples_leaf=1):
        assert (parent is None) or isinstance(parent, TreeNode)
        assert min_samples_leaf >= 1
        assert x.shape[0] == y.shape[0]

        self.parent = parent
        self.is_root = parent is None

        self.children: List[TreeNode] = []
        self.depth = 0

        self.x_train = x  # features
        self.y_train = y  # labels (or targets/classes)
        self.num_samples = self.x_train.shape[0]

        # compute split range
        self.x_low = np.min(self.x_train, axis=0)
        self.x_range = np.max(self.x_train, axis=0) - self.x_low

        self.split_index = -1.0
        self.split_values = None
        self.all_split_values = None  # = `split_values` with [-inf, +inf]

        if self.is_root:
            self.classes, self.class_counts = self._find_classes_and_counts()
            self.min_samples_leaf = int(min_samples_leaf)
        else:
            self.classes = self.parent.classes
            self.class_counts = self._compute_class_counts()
            self.min_samples_leaf = self.parent.min_samples_leaf

        self.is_leaf = self._check_if_leaf()
        self.impurity = self._compute_gini()

    def __str__(self):
        if self.is_leaf:
            return f'Leaf@{self.depth} {self.class_counts}'

        elif self.is_root:
            name = 'Root'
        else:
            name = f'Node@{self.depth}'

        if self.split_values is None or utils.is_empty(self.split_values):
            split = '()'

        elif len(self.split_values) == 1:
            split = f'x[{self.split_index}] <= {str(round(self.split_values[0], 2))}'
        else:
            a = round(self.split_values[0], 2)
            b = round(self.split_values[1], 2)
            split = f'{str(a)} <= x[{self.split_index}] <= {str(b)}'

        return f'{name} [{split} -> {self.class_counts}]'

    # TODO: rename to `grow`?
    def split(self, index: int, values) -> bool:
        """Grows the tree"""
        # values = np.reshape(values, newshape=[-1])
        # values = np.sort(values)

        if self.x_range[index] == 0.0:
            return False

        values = self.x_low[index] + (values * self.x_range[index])
        values = np.concatenate([-self.INF, values, self.INF], axis=0)

        for i in range(values.shape[0] - 1):
            x = self.x_train[:, index]
            mask = (x > values[i]) & (x <= values[i + 1])

            if np.sum(mask) == 0:
                return False

            node = TreeNode(x=self.x_train[mask], y=self.y_train[mask], parent=self)
            node.depth = self.depth + 1

            self.children.append(node)
            # print(f'--> [{i+1}/{values.shape[0]-1}] append {node}')

        self.split_index = index
        self.split_values = values[1:-1]
        self.all_split_values = values

        return True

    # def can_split(self, index: int) -> bool:
    #     if self.x_range[index] == 0.0:
    #         return False
    #
    #     return True

    def is_empty(self) -> bool:
        """Check whether the tree is empty or not"""
        return self.is_root and len(self.children) == 0

    def prob(self) -> np.ndarray:
        sum_counts = np.sum(self.class_counts)

        if sum_counts == 0.0:
            return np.zeros_like(self.class_counts)

        return self.class_counts / sum_counts

    def has_children(self) -> bool:
        """Check whether the node has at least one child or not"""
        return len(self.children) > 0

    def print(self):
        """Prints the tree structure, the splitting rules, and the sample counts"""
        queue = [(self, '')]

        while len(queue) > 0:
            node, space = queue.pop(0)
            print(f'{space}{node}')

            # for child in node.children:
            #     queue.append((child, f'{space}\t'))

            for child in reversed(node.children):
                queue.insert(0, (child, f'{space}\t'))

    def stringify(self) -> str:
        """Returns a string representation from the current node up to the parent node"""
        node = self
        string = str(self)

        while node.parent:
            node = node.parent
            string = f'{str(node)} {string}'

        return string

    def as_features(self, split_size: int) -> np.ndarray:
        """Transforms the entire sub-tree from the current node up to the root as a matrix of features.
            - Must be called from a non-leaf node.
        """
        assert not self.is_leaf

        node = self
        features = [self.to_vector(split_size)]

        while node.parent:
            node = node.parent
            features.insert(0, node.to_vector(split_size))

        return np.array(features, dtype=np.float32)

    # TODO: use "impurity" as feature?
    # TODO: add "is_leaf" or substitute with "num_children"
    def to_vector(self, split_size: int) -> np.ndarray:
        """Transforms the current node to a vector of features:
            - 3: depth, split_index, num children,
            - N: split_values,
            - M: class_counts.
        """
        if not self.has_children():
            split_values = np.zeros(shape=(split_size,), dtype=np.float32)

        elif len(self.split_values) < split_size:
            # pad with trailing zeros
            split_values = np.zeros(shape=(split_size,), dtype=np.float32)
            split_values[:len(self.split_values)] = self.split_values
        else:
            split_values = self.split_values

        features = np.concatenate([
            # [self.depth, self.split_index, len(self.children)],  # TODO: divide by their max
            [self.depth, self.split_index / self.x_train.shape[-1], len(self.children) / split_size],
            split_values,
            # self.class_counts,
            self.prob()  # class prob
        ])

        return np.asarray(features, dtype=np.float32)

    def _compute_gini(self):
        # source: https://scikit-learn.org/stable/modules/tree.html#classification-criteria
        proportion = self.class_counts / self.class_counts.sum()
        return np.sum(proportion * (1.0 - proportion))

    def _compute_entropy(self):
        # source: https://scikit-learn.org/stable/modules/tree.html#classification-criteria
        proportion = self.class_counts / self.class_counts.sum()
        return -np.sum(proportion * np.log(proportion))

    def _compute_misclassification(self):
        # source: https://scikit-learn.org/stable/modules/tree.html#classification-criteria
        proportion = self.class_counts / self.class_counts.sum()
        return 1.0 - np.max(proportion)

    def _find_classes_and_counts(self):
        y_train = self.y_train

        if len(y_train.shape) == 1 or y_train.shape[-1] == 1:
            return np.unique(y_train, return_counts=True)

        # assume `y_train` is one-hot encoded
        classes = np.arange(y_train.shape[-1])
        counts = np.sum(y_train, axis=0)

        return classes, counts

    def _compute_class_counts(self) -> np.ndarray:
        if len(self.y_train.shape) == 1 or self.y_train.shape[-1] == 1:
            return np.asarray([np.sum(self.y_train == c) for c in self.classes])

        return np.sum(self.y_train, axis=0)

    def _check_if_leaf(self) -> bool:
        """A node is a leaf if either:
            1. the number of samples is <= `min_samples_leaf`, or
            2. the node is pure, i.e. only one class count is non-zero: e.g. [0, 10, 0]
        """
        few_samples = np.all(self.class_counts <= self.min_samples_leaf)  # 1
        # few_samples = self.class_counts.sum() <= self.min_samples_leaf  # 1
        is_pure = np.sum(self.class_counts == 0) == len(self.classes) - 1  # 2

        return is_pure or few_samples


class TreeClassifier(TreeNode):
    """Decision tree for classification problems"""
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, max_depth=None, max_split_values=2, min_samples_leaf=1):
        assert max_split_values >= 1
        assert min_samples_leaf >= 1
        assert x_train.shape[0] == y_train.shape[0]

        super().__init__(x=x_train, y=y_train, min_samples_leaf=int(min_samples_leaf))

        if isinstance(max_depth, (int, float)):
            assert max_depth >= 1
            self.max_depth = int(max_depth)
        else:
            # TODO: remove inf depth
            self.max_depth = np.inf

        self.max_split = int(max_split_values)

    # TODO: provide keras-style progress bar
    def predict(self, x, batch_size: int, already_batched=False, prefetch=2, **kwargs):
        assert batch_size >= 1

        if not already_batched:
            import tensorflow as tf

            x = tf.data.Dataset.from_tensor_slices(x)
            x = x.batch(int(batch_size))
            x = x.prefetch(buffer_size=int(prefetch))

        out = []

        for batch in x:
            probs = self.predict_batch(batch, **kwargs)
            out.append(probs)

        return np.concatenate(out, axis=0)

    def predict_batch(self, batch, sort=True, debug=False):
        batch = np.asarray(batch, dtype=np.float32)
        assert not utils.is_empty(batch)

        if len(batch.shape) == 1:
            batch = np.reshape(batch, newshape=(1, -1))
        else:
            assert len(batch.shape) == 2

        # queue: node, data, indices (for sorting)
        queue = [(self, batch, np.arange(batch.shape[0]))]
        probs = []
        indices = []

        while len(queue) > 0:
            node, x, idx = queue.pop(0)
            if debug:
                print('POP', str(node), len(queue), f'child: {len(node.children)}')

            if node.is_leaf or len(node.children) == 0:
                # repeat `prob` x.shape[0] times
                if debug:
                    print(node.class_counts)

                prob = np.repeat(node.prob()[None, :], repeats=x.shape[0], axis=0)
                probs.append(prob)
                indices.append(idx)
                continue

            features = x[:, node.split_index]

            for i in range(len(node.all_split_values) - 1):
                lower = node.all_split_values[i]
                upper = node.all_split_values[i + 1]

                mask = (features > lower) & (features <= upper)
                data = x[mask]

                if data.shape[0] > 0:
                    queue.append((node.children[i], data, idx[mask]))

                    if debug:
                        print('\tadded', i)

        probs = np.concatenate(probs, axis=0)

        if sort:
            indices = np.concatenate(indices, axis=0)
            indices = np.sort(indices)
            return probs[indices]

        return probs

    def total_impurity(self) -> float:
        impurity = 0.0
        queue = [self]

        while len(queue) > 0:
            node = queue.pop(0)

            if node.is_leaf or len(node.children):
                impurity += node.impurity
            else:
                queue.extend([child for child in node.children])

        return impurity


class TreeRegressor:
    """Decision tree for regression problems"""
    pass
