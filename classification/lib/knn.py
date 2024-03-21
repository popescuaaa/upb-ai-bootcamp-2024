"""
KNN implementation

author: Andrei Gabriel Popescu
email: andrei.popescu@orange.com

@credits
The classifier was insipired by cs231n homework assignment from Stanford.

@Tasks

1. Complete the missing parts in the code bellow in order to succesfully run the classifier
2. Explore different alternatives for computing the best k value

"""

from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from builtins import object


class KNN(object):
    def __init__(self) -> None:
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = np.array([])
        self.y_train = np.array([])


    def set_values(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Memorization step
        """
        self.X_train = X
        self.y_train = y

    def  predict(self, X: np.ndarray, loops: int = 0, k: int = 1) -> None:
        
        if loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif loops == 2:
            dists = self.compute_distances_two_loops(X)
        elif loops == 3:
            dists = self.compute_distances_theree_loops(X)
        else:
            raise ValueError("Invalid value {} for number of loops. The value should be 0, 1, 2 or 3.".format(loops))

        print("dist", dists)
        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X: np.ndarray) -> np.ndarray:
        """
        TODO 4:                                                             
        Compute the l2 distance between all test points and all training    
        points without using any explicit loops, and store the result in    
        dists.                                                              
                                                                        
        You should implement this function using only basic array operations;
        in particular you should not use functions from scipy,              
        nor use np.linalg.norm().                                           
                                                                        
        HINT: Try to formulate the l2 distance using matrix multiplication  
              and two broadcast sums.                                       

        """

        # print("train", self.X_train.T.shape)
        # print("X", X.shape)

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        term_1 = np.sum(X ** 2, axis=1)
        # print(term_1.shape)
        term_2 = -2 * np.matmul(X, self.X_train.T)
        term_3 = np.sum(self.X_train ** 2, axis=1)
        dists = np.sqrt(term_1.reshape(-1, 1) + term_2 + term_3)

        return dists


    def compute_distances_one_loop(self, X: np.ndarray) -> np.ndarray:
        """
        TODO 3:
                                                                       
        Compute the l2 distance between the ith test point and all training 
        points, and store the result in dists[i, :].                        
        Do not use np.linalg.norm().                                       

        X[i] - X_train would have the shape (d) - (num_train, d), which
        is not a valid operation (they need to have the same shape).
        
        Since we don't want to iterate each row of X_train, one solution
        might be to convert X[i] to a matrix with num_train rows, where
        each row is a copy of X[i]
        
        We can use achieve this using np.tile(X[i], num_train, 1),
        however numpy does broadcasting by default, so you can actually
        subtract X[i] - X_train and it will all work out nicely

        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis=1))

        return dists
    
    def compute_distances_two_loops(self, X: np.ndarray) -> np.ndarray:
        """
        TODO 2:                                                            
        Compute the l2 distance between the ith test point and the jth 
        training point, and store the result in dists[i, j]. You should
        not use a loop over dimension, nor use np.linalg.norm().
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
        
        return dists
    
    def compute_distances_theree_loops(self, X: np.ndarray) -> np.ndarray:
        """
        # TODO 1: Complete this.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dim = X.shape[1]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                for k in range(dim):
                    dists[i][j] += (X[i][k] - self.X_train[j][k]) ** 2
                dists[i][j] = np.sqrt(dists[i][j])

        return dists

    def predict_labels(self, dists: np.ndarray, k: int = 1):
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
            closest_y = []
            indices = np.argsort(dists[i])[:k]
            closest_y = self.y_train[indices]
            unique, counts = np.unique(closest_y, return_counts=True)
            max_indices = np.argwhere(counts == np.max(counts))
            y_pred[i] = unique[np.min(max_indices)]
        return y_pred

