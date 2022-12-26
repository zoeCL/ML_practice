import os
import itertools
import numpy as np
import _pickle as cPickle
import sklearn.preprocessing

class rl():
    def __init__(self, len_action_space, len_feature):
        self.len_action_space = len_action_space
        self.weights = np.random.random_sample((len_feature, self.len_action_space))
        self.total_rewards = []
        self.feature_prev = None
        self.prediction_prev = None
        self.action_prev = None
        self.temperature = 2.0

    def rl_init(self, feature):
        feature = np.array(feature)
        self.feature_prev = feature
        self.prediction_prev = self.prediction(self.feature_prev)
        self.action_prev = None

    def standardize_data(self, data):
        """
        scaler is fixed to StandardScaler: removing the mean and scaling data into unit variance
        """
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(data)
        return scaler.transform(data)

    def prediction(self, feature):
        """
        Returns: q values for all actions
        """
        predictions = np.zeros((self.len_action_space, 1))
        scaled_feature = feature
        for i in range(self.len_action_space):
            predictions[i] = np.transpose(scaled_feature).dot(self.weights[:, i])
        return predictions.flatten()

    def make_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
        Returns: action to pick which followed epsilon greedy policy
        """
        softmax_predictions = self.softmax(self.prediction_prev)
        self.action_prev = np.random.choice(range(len(softmax_predictions)), 1, p=softmax_predictions.tolist())[0]
        return self.action_prev  # this is the index of the action in action_space

    def softmax(self, x):
        x = x / self.temperature
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def q_learning(self, feature_aft, reward, discount_factor=0.9, alpha=0.01):
        """
        Q-learning: algorithm for off-policy TD control using Function Approximation
        """
        feature_aft = np.array(feature_aft)
        self.total_rewards.append(reward)
        prediction_aft = self.prediction(feature_aft)
        self.weights[:, self.action_prev] = self.weights[:, self.action_prev] - alpha * (- reward - discount_factor * self.weights[:, self.action_prev].dot(feature_aft) + self.weights[:, self.action_prev].dot(self.feature_prev)) * self.feature_prev
        self.feature_prev = feature_aft
        self.prediction_prev = prediction_aft

    def save_weights(self, fold_path):
        if not os.path.exists(fold_path):
            print ('[ERROR] The output path %s does not exist' % (fold_path))
            return
        else:
            file_path = os.path.join(fold_path, "weights.pkl")
            with open(file_path, 'wb') as writefile:
                cPickle.dump(self.weights, writefile)
            print ('[INFO] Sucessfully save weights file to %s' % (file_path))

    def load_weights(self, fold_path):
        file_path = os.path.join(fold_path, "weights.pkl")
        if not os.path.exists(file_path):
            print ('[ERROR] The weights file %s does not exist' % (fold_path))
            return
        else:
            with open(file_path, 'rb') as readfile:
                self.weights = cPickle.load(readfile)
            print ('[INFO] Sucessfully load weights file from %s' % (file_path))