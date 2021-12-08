import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

DATA_FILE = 'data_banknote_authentitcation.txt'
RANDOM_SEED = 9248
RUNS = 5
EPOCHS = 1000
HIDDEN_LAYER_SIZES = [2, 3, 4, 5]


def ReLU(m):
    m[m < 0] = 0
    return m


"""
Misclassification error and plotting
"""


def misclassification(model, X, t):
    test_values = model.forwardPass(X.T)["l3_out"].T
    return np.sum(np.rint(test_values) != t) / t.shape[0]


def plot(title, model, X_train, t_train, X_val, t_val, X_test, t_test):

    trainingErrors = []
    valErrors = []
    misclassificationRates = []

    for i in range(EPOCHS):

        misclassificationRates.append(misclassification(model, X_test, t_test))

        error = model.error(X_val, t_val)
        valErrors.append(error)

        train_values = model.forwardPass(X_train.T)
        error = model.error(X_train, t_train)
        trainingErrors.append(error)

        grads = model.computeGradients(train_values)
        model.updateWeights(grads)

    plt.figure()
    plt.plot(trainingErrors, label='Training Errors')
    plt.plot(valErrors, label='Validation Errors')
    plt.plot(misclassificationRates, label='Misclassification rate')
    plt.title(title)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()


"""
Neural Network class:
- processess all NN operations from training to back prop and getting errors

"""


class NeuralNetwork():

    def __init__(self, X, t, HL1_numNeurons, HL2_numNeurons, alpha):
        self.alpha = alpha
        self.X = X
        self.t = t
        self.numFeatures = X.shape[1]
        self.w1 = np.random.randn(HL1_numNeurons, self.numFeatures)
        self.w2 = np.random.randn(HL2_numNeurons, HL1_numNeurons)
        self.w3 = np.random.randn(t.shape[1], HL2_numNeurons)
        self.HL1_numNeurons = HL1_numNeurons
        self.HL2_numNeurons = HL2_numNeurons

    # train
    def train(self):
        for _ in range(EPOCHS):
            self.updateWeights(self.computeGradients(
                self.forwardPass(self.X.T)))

    # forward propogation
    def forwardPass(self, X):
        res = {}
        res["l1_z"] = np.dot(self.w1, X)
        res["l1_out"] = ReLU(res["l1_z"])
        res["l2_z"] = np.dot(self.w2, res["l1_out"])
        res["l2_out"] = ReLU(res["l2_z"])
        res["l3_z"] = np.dot(self.w3, res["l2_out"])
        res["l3_out"] = ReLU(res["l3_z"])
        return res

    # back propogation
    def computeGradients(self, forwardPassResults):
        res = {}
        m_inv = 1 / len(self.t)
        t = self.t.T
        X = self.X.T

        dA = m_inv * (forwardPassResults["l3_out"] - t)
        dZ = dA
        res["w3"] = m_inv * np.dot(dZ, forwardPassResults['l2_out'].T)

        dA = np.dot(self.w3.T, dZ)
        dZ = np.multiply(dA, np.where(forwardPassResults["l2_out"] > 0, 1, 0))
        res["w2"] = m_inv * np.dot(dZ, forwardPassResults['l1_out'].T)

        dA = np.dot(self.w2.T, dZ)
        dZ = np.multiply(dA, np.where(forwardPassResults["l1_out"] > 0, 1, 0))
        res["w1"] = m_inv * np.dot(dZ, X.T)

        return res

    # gradient descent from lectures
    def updateWeights(self, grads):
        self.w1 -= self.alpha * grads["w1"]
        self.w2 -= self.alpha * grads["w2"]
        self.w3 -= self.alpha * grads["w3"]

    def costFn(self, y, t):
        """ 
        summation = 0
        for i in range(len(t_actual)):
            summation += t_actual[i][0] * np.logaddexp(0,-t_predicted[i][0]) + (1-t_actual[i][0]) * np.logaddexp(0, t_predicted[i][0])
        """
        cost = 0
        N = len(t)
        for i in range(N):
            cost += t[i][0] * np.logaddexp(0, -y[i][0]) + \
                (1-t[i][0]) * np.logaddexp(0, y[i][0])

        return cost / N

    def error(self, X, t):
        forwardPass = self.forwardPass(X.T)
        return self.costFn(forwardPass["l3_out"].T, t)


"""
main function:

- first processes and splits the data
- then it controls the different trials of each set of parameters
- and then it plots the best models from each number of features


"""
if __name__ == '__main__':
    # Process all the data
    sc = StandardScaler()
    data = np.loadtxt(fname="data_banknote_authentication.txt", delimiter=',')
    split = 0.5
    val_split = 0.25

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(data)

    length = data.shape[0]
    train_data = data[:int(length * split)]
    val_data = data[int(length * split):int(length *
                                            split) + int(length * val_split)]
    test_data = data[int(length * split) + int(length * val_split):]

    numFeatures = data.shape[1] - 1
    X_train, t_train = train_data[:, :numFeatures], train_data[:, numFeatures]
    X_val, t_val = val_data[:, :numFeatures], val_data[:, numFeatures]
    X_test, t_test = test_data[:, :numFeatures], test_data[:, numFeatures]

    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    t_train = np.expand_dims(t_train, axis=1)
    t_test = np.expand_dims(t_test, axis=1)
    t_val = np.expand_dims(t_val, axis=1)

    featuresArr = [2, 3, 4]

    global_best_err = math.inf
    global_best_model = None

    for i in featuresArr:
        best_err = math.inf
        best_model = None
        print("\n##########################")
        print("number of features = ", i)
        print("##########################")
        for l1_size in HIDDEN_LAYER_SIZES:
            for l2_size in HIDDEN_LAYER_SIZES:
                err = 0
                for _ in range(RUNS):
                    currModel = NeuralNetwork(
                        X_train[:, :i], t_train, l1_size, l2_size, 0.03)
                    currModel.train()
                    err += currModel.error(X_val[:, :i], t_val)/RUNS

                # print("\n-----------------")
                #print("hidden layer 1 size =",l1_size)
                #print("hidden layer 2 size =",l2_size)
                # print("-----------------")

                if err < best_err:
                    best_err = err
                    best_model = currModel

                if err < global_best_err:
                    global_best_err = err
                    global_best_model = currModel

                print(l1_size, l2_size, err, misclassification(
                    currModel, X_train[:, :i], t_train))

                #print("error for this configuration =", err)
                #print("misclass rate for this configuration =", misclassification(currModel,X_train[:,:i],t_train))

        plot(str("Num Features =" + str(i) + ", HL1 Size/HL2 Size = " + str(best_model.HL1_numNeurons) + ', ' + str(best_model.HL2_numNeurons)),
             best_model,
             X_train[:, :i],
             t_train,
             X_val[:, :i],
             t_val,
             X_test[:, :i],
             t_test)

    print("\n#######################################")
    print("# Best Model Parameters:")
    print("# Number of features =", global_best_model.numFeatures)
    print("# Hidden Layer 1 Number of Neurons =",
          global_best_model.HL1_numNeurons)
    print("# Hidden Layer 2 Number of Neurons =",
          global_best_model.HL2_numNeurons)
    print("# Best Model Test Error =",  global_best_model.error(
        X_test[:, :global_best_model.numFeatures], t_test))
    print("# Best Model Validation CE Error =", global_best_err)
    print("#######################################")
