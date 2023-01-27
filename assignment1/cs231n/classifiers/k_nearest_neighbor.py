from builtins import range
from builtins import object 
import numpy as np
from past.builtins import xrange
from typing import Tuple

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (num_train,) containing the training labels, where
             y[i] is the label for X[i].
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        if (not isinstance(k, int)) or (k < 1):
            raise ValueError("Invalid value %d for k" % k)
        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) # initialize a numpy array
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                dists[i, j] = np.sqrt(np.sum((self.X_train[j] - X[i])**2)) # L2 distance

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dists[i] = np.sqrt(np.sum((X[i] - self.X_train)**2, axis=1))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # X.shape=(num_test, D); self.X_train.shape=(num_train, D)
        # when X.shape=(num_test, 1, D), X and self.X_train can be broadcasted as
        # -> X.shape=(num_test, num_train, D)
        # -> self.X_train.shape=(num_test, num_train, D)
        # hence we can substract X and self.X_train
        # but this implementation may run out of memory
        # dists = np.sqrt(np.sum((np.expand_dims(X, axis=1) - self.X_train)**2, axis=2))
        
        # the square of l2 distance of two vectors x and y ||x-y||_2^2 is
        # np.dot(x-y, x-y) = np.dot(x,x) + np.dot(y,y) - 2*np.dot(x,y)
        # aka (x-y)^2 = x^2 + y^2 - 2xy
        # to implement fully vectorization, we need to write a formula
        # which can compute l2 distance of any two vectors at once
        # refer to https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        x2 = np.sum(X**2, axis=1) # shape=(num_test,)
        y2 = np.sum(self.X_train**2, axis=1) # shape=(num_train,)
        xy = X @ self.X_train.T # @: matrix multiplication operator
        x2 = np.expand_dims(x2, axis=1) # shape=(num_test, 1) for broadcast
        sqr_dists = x2 + y2 - 2*xy # shape=(num_test, num_train)
        dists = np.sqrt(sqr_dists)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            # closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # the index of k nearest neighbor in X_train
            closest_pts: np.ndarray = np.argsort(dists[i])[:k] # 1D
            # get the label of k nearest neighbor
            closest_y: np.ndarray = self.y_train[closest_pts] # 1D

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            if k == 1:
                y_pred[i] = closest_y.item()
            else:
                # get common y and their frequency
                vals, counts = np.unique(closest_y, return_counts=True)
                # get the index of most common y
                most_common_y_inds: np.ndarray = np.where(counts == max(counts))[0] # 1D
                
                if most_common_y_inds.size > 1: # more than 1 most common y
                    y_pred[i] = min(vals[most_common_y_inds]) # break ties
                else:
                    y_pred[i] = vals[most_common_y_inds].item()
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
