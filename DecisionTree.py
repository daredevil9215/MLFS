import numpy as np
from collections import Counter

def entropy(y: np.ndarray) -> float:
    """
    Calculates the entropy of target values.

    Parameters
    ----------
    y : numpy.ndarray
        Target values of shape (n_samples,).

    Returns
    -------
    e : float
        Entropy associated with the given target values.
    """
    # Get the probability for each value
    probabilites = np.bincount(y) / len(y)

    # Get the number of unique values (classes)
    classes = len(set(y))

    if classes != 1:
        return -1 * np.sum([p * np.log(p) / np.log(classes) for p in probabilites if p > 0])
    else:
        return 0

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None) -> None:
        """
        Node for building a decision tree. 
        Each node is a point at which a decision is made.

        Attributes
        ----------
        feature : int
            Index of the feature column which is used as the feature space for the node.

        threshold : float
            Value that best splits the feature space based on information gain.

        left : Node
            Subset of feature space where the values are below or equal to the threshold.

        right: Node
            Subset of feature space where the values are above the threshold.

        value : int
            The target value of the most common class of a node if the stopping criteria is reached.
            If the node has a value then it is a leaf node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        """
        Method for determining whether a node is a leaf node.
        Node is a leaf node only if it has a value assigned to it.

        Returns
        -------
        result : bool
        """
        return self.value != None
    
class DecisionTree:

    def __init__(self, min_samples_split: int = 2, max_depth: int = 100, n_feats: int = None) -> None:
        """
        Decision Tree classifier.

        Parameters
        ----------
        min_samples_split : int, default=2
            Minimum number of samples required to split a node.

        max_depth : int, default=100
            Maximum depth of the tree.

        n_feats : int, default=None
            Number of features to choose from when performing a node split.

        Attributes
        ----------
        root : Node
            Root node of the tree.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build a decision tree.

        Parameters
        ----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).

        y : numpy.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        None
        """
        # Set n_feats
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        # Create root node
        self.root = self._grow_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the decision tree.

        Parameters
        ----------
        X : numpy.ndarray
            Samples of shape (n_samples, n_features).

        Returns
        -------
        y : numpy.ndarray
            Predicted values of shape (n_samples,).
        """
        # Traverse the tree
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        """
        Helper method for traversing the decision tree recursively.

        Parameters
        ----------
        x : np.ndarray
            Sample of shape (n_features,).

        node : Node
            Node at which the traversal starts.

        Returns
        -------
        value : int
            Value located at the leaf node.
        """
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Helper method for growing the decision tree recursively.

        Parameters
        ----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).

        y : numpy.ndarray
            Target values of shape (n_samples,).

        depth : int, default=0
            Current depth of tree.

        Returns
        -------
        node : Node
            Node containing children nodes or a leaf node if the stopping criteria is reached.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria (leaf node reached)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Randomize features
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Search for best information gain and save best split feature and best split threshold
        best_feature, best_threshold = self._best_criteria(X, y, feat_idxs)

        # Split the current node into left and right nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feature, best_threshold, left, right)
        
    def _most_common_label(self, y: np.ndarray) -> int:
        """
        Helper method for determining the most common label in an array.

        Parameters
        ----------
        y : np.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        label : int
            Most common target value.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _best_criteria(self, X: np.ndarray, y:np.ndarray, feat_idxs: np.ndarray) -> tuple[int, float]:
        """
        Helper method for determining the best criteria for a node split (feature index and feature threshold).

        Parameters
        ----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).

        y : numpy.ndarray
            Target values of shape (n_samples,).

        feat_idxs : numpy.ndarray
            Array of feature indices.

        Returns
        -------
        criteria : tuple[int, float]
            Best feature index and best feature threshold.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        # Loop through all feature indices
        for feat_idx in feat_idxs:

            # Select the feature column
            X_column = X[:, feat_idx]
            # Get the unique column values to be thresholds
            thresholds = np.unique(X_column)

            # Loop through all thresholds
            for threshold in thresholds:
                
                # Calculate the information gain for the given target values, column and threshold
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    # Store the new best gain, feature index and feature threshold
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold
    
    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, split_threshold: float) -> float:
        """
        Helper method for calculating information gain.

        Parameters
        ----------
        y : numpy.ndarray
            Target values of shape (n_samples,).
        
        X_column : numpy.ndarray
            Feature column of shape (n_samples,).

        split_threshold : float
            Threshold used for splitting the feature column.

        Returns
        -------
        information_gain : float
        """
        # Calculate entropy of the target values
        parent_entropy = entropy(y)

        # Split the feature column based on the given threshold
        left_idxs, right_idxs = self._split(X_column, split_threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        # Calculate left and right child entropies
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        # Calculate weighted average of left and right child entropies
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _split(self, X_column: np.ndarray, split_threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper method for splitting the values based on threshold.

        Parameters
        ----------
        X_column : numpy.ndarray
            Feature column of shape (n_samples,).

        split_threshold : float
            Threshold used for splitting the values.

        Returns
        -------
        indices : tuple[numpy.ndarray, numpy.ndarray]
            Left and right indices.
            Left indices are the indices of values below or equal to the threshold.
            Right indices are the indices of values above the threshold.
        """
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs