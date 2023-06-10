import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot(data, labels, w):
    fig, ax = plt.subplots()

    c0 = data[labels == 0]
    c1 = data[labels == 1]

    ax.scatter(c0[:, 0], c0[:, 1], c='red')
    ax.scatter(c1[:, 0], c1[:, 1], c='blue')

    a, b, c = w
    m = -a / b
    b = -c / b

    x = np.arange(np.min(data[:, 0]), np.max(data[:, 0]), 0.1)
    y = m * x + b
    plt.plot(x, y)

    plt.show()


def load_data(file_path):
    """
    Load data, split to features and label, standardization of the data, and add '1' to each feature to build later the w include bias
    :param file_path: path to excel file
    :return: np array of standardized features and their labels
    """
    # Load data from CSV file
    data_path = open(file_path, 'r')
    data = np.loadtxt(data_path, delimiter=",")
    # split data to features and labels
    y = data[:, -1].astype(int)  # labels
    x = data[:, :-1].astype(float).astype(float)  # features
    # Data standardization
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / std
    #  Add columns of 1
    new_column = np.ones((x.shape[0], 1))
    x = np.concatenate((x, new_column), axis=1)
    return x, y


def logistic_regression_via_GD(P, y, lr, epochs=1000):
    """
    :param epochs: number of iteration for train the model
    :param P: np array of ‘n’ rows and ‘d’ columns
    :param y: a label vector of ‘n’ entries
    :param lr: learning rate parameter
    :param epochs: number of training iterations
    :return: vector ‘w’ (and ‘b’) which minimizes the logistic regression cost function on ‘P’ and ‘y’
    """
    number_of_samples, number_of_features = P.shape
    w = np.zeros(number_of_features)
    for _ in range(epochs):
        z = np.dot(w.transpose(), P.transpose())
        # Get predictions with sigmoid(z)
        probabilities = 1 / (1 + np.exp(-z))
        # Calculate partial derivatives (gradient descent)
        dw = (1 / number_of_samples) * np.dot(P.T, (probabilities - y))
        # Update w
        w -= lr * dw
    return w


def predict(w, p):
    """
    :param w: pretrained weights
    :param p:an input vector (numpy) which represents a sample
    :return:he class prediction for ‘p’ of the logistic regression model defined by ‘w’ and ‘b’.
    """
    z = np.dot(p, w.transpose())
    probability = 1 / (1 + np.exp(-z))
    prediction = (probability > 0.5).astype(int)
    return prediction


# load data
x, y = load_data('exams.csv')

# Split data for model training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# train the model
w = logistic_regression_via_GD(x_train, y_train, 0.1)

accuracies = [0.0] * 1000
for i in range(1000):
    correct = 0
    # Split data for model evaluation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    for j, sample in enumerate(x_test):
        prediction = predict(w, sample)
        correct += (prediction == y_test[j])
    accuracies[i] = correct / len(y_test)

# plot(x[:, :-1], y, w)
accuracy = sum(accuracies) / len(accuracies)
print(f"Avg test accuracy: {accuracy * 100}%")
