import numpy as np
import pvml
import matplotlib.pyplot as plt
import sys


def multinomial_naive_bayes_train(X, Y):
    m, n = X.shape
    k = Y.max() + 1
    k = k.as_integer_ratio()[0]  # transfer numpy.float64 to int
    probs = np.empty((k, n))
    for c in range(k):
        counts = X[Y == c, :].sum(0)
        tot = counts.sum()
        probs[c, :] = (counts + 1) / (tot + n)
    priors = np.bincount(Y.astype('int')) / m
    W = np.log(probs).T
    b = np.log(priors)
    return W, b


def multinomial_naive_bayes_inference(X, W, b):
    scores = X @ W + b.T
    labels = np.argmax(scores, 1)
    return labels


def naive_bayes(Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation):
    w, b = multinomial_naive_bayes_train(Xtrain, Ytrain)
    predictions = multinomial_naive_bayes_inference(Xtrain, w, b)
    accuracy_train = (predictions == Ytrain).mean()
    show_confusion_matrix(Ytrain, predictions)
    plt.show()

    w, b = multinomial_naive_bayes_train(Xtest, Ytest)
    predictions = multinomial_naive_bayes_inference(Xtest, w, b)
    accuracy_test = (predictions == Ytest).mean()
    show_confusion_matrix(Ytest, predictions)
    plt.show()

    w, b = multinomial_naive_bayes_train(Xvalidation, Yvalidation)
    predictions = multinomial_naive_bayes_inference(Xvalidation, w, b)
    accuracy_validation = (predictions == Yvalidation).mean()
    show_confusion_matrix(Yvalidation, predictions)
    plt.show()

    print("----------------------")
    print("Accuracies for Naive Bayes Classifier:")
    print("Training accuracy:", accuracy_train * 100)
    print("Testing accuracy:", accuracy_test * 100)
    print("Validation accuracy:", accuracy_validation * 100)


def gaussian_naive_bayes_train(X, Y):
    k = (Y.max() + 1).astype('int')
    m, n = X.shape
    means = np.empty((k, n))
    vars = np.empty((k, n))
    priors = np.bincount(Y.astype('int')) / m
    for c in range(k):
        means[c, :] = X[Y == c, :].mean(0)
        vars[c, :] = X[Y == c, :].var(0)
    return means, vars, priors


def gaussian_naive_bayes_inference(X, means, vars, priors):
    m = X.shape[0]
    k = means.shape[0]
    scores = np.empty((m, k))
    for c in range(k):
        diffs = ((X - means[c, :]) ** 2) / (2 * vars[c, :])
        scores[:, c] = - diffs.sum(1)
    scores -= 0.5 * np.log(vars).sum(1)
    scores += np.log(priors)
    labels = np.argmax(scores, 1)
    return labels


def gaussian_naive_bayes(Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation):
    means, vars, priors = gaussian_naive_bayes_train(Xtrain, Ytrain)
    predictions = gaussian_naive_bayes_inference(Xtrain, means, vars, priors)
    accuracy_train = (predictions == Ytrain).mean()
    show_confusion_matrix(Ytrain, predictions)
    plt.show()

    means, vars, priors = gaussian_naive_bayes_train(Xtest, Ytest)
    predictions = gaussian_naive_bayes_inference(Xtest, means, vars, priors)
    accuracy_test = (predictions == Ytest).mean()
    show_confusion_matrix(Ytest, predictions)
    plt.show()

    means, vars, priors = gaussian_naive_bayes_train(Xvalidation, Yvalidation)
    predictions = gaussian_naive_bayes_inference(Xvalidation, means, vars, priors)
    accuracy_validation = (predictions == Yvalidation).mean()
    show_confusion_matrix(Yvalidation, predictions)
    plt.show()

    print("----------------------")
    print("Accuracies for Gaussian Naive Bayes Classifier:")
    print("Training accuracy:", accuracy_train * 100)
    print("Testing accuracy:", accuracy_test * 100)
    print("Validation accuracy:", accuracy_validation * 100)


# MEASURE THE ACCURACY ON A GIVEN SET
def accuracy(net, X, Y):
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100


# MinMax Normalization
def minmax_normalization(Xtrain, Xtest, Xvalidate):
    xmin = Xtrain.min(0)
    xmax = Xtrain.max(0)
    Xtrain = (Xtrain - xmin) / (xmax - xmin)
    Xtest = (Xtest - xmin) / (xmax - xmin)
    Xvalidate = (Xvalidate - xmin) / (xmax - xmin)
    return Xtrain, Xtest, Xvalidate


# Mean-Variance Normalization
def meanvar_normalization(Xtrain, Xtest, Xvalidate):
    mu = Xtrain.mean(0)
    std = Xtrain.std(0)
    std = np.maximum(std, 1e-15)

    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
    Xvalidate = (Xvalidate - mu) / std

    return Xtrain, Xtest, Xvalidate


# MAX-Absolute Normalization
def maxabs_normalization(Xtrain, Xtest, Xvalidate):
    amax = np.abs(Xtrain).max(0)
    Xtrain = Xtrain / amax
    Xtest = Xtest / amax
    Xvalidate = Xvalidate / amax
    return Xtrain, Xtest, Xvalidate


# L2 Normalization
def l2_normalization(X):
    q = np.sqrt((X ** 2).sum(1, keepdims=True))
    q = np.maximum(q, 1e-15)  # 1e -15 avoids division by zero
    X = X / q
    return X


# CREATE AND TRAIN THE MULTI-LAYER PERCEPTRON
def multi_layer_perceptron(Xtrain, Ytrain, Xtest, Ytest):
    net = pvml.MLP([2000, 8])
    m = Ytrain.size
    plt.ion()
    train_accs = []
    test_accs = []
    epochs = []
    batch_size = 10
    for epoch in range(10):
        net.train(Xtrain, Ytrain.astype('int'), 1e-4, steps=m // batch_size, batch=batch_size)
        if epoch % 5 == 0:  # to speed up
            train_acc = accuracy(net, Xtrain, Ytrain)
            test_acc = accuracy(net, Xtest, Ytest)
            print(epoch, train_acc, test_acc)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            epochs.append(epoch)
            plt.clf()
    plt.plot(epochs, train_accs)
    plt.plot(epochs, test_accs)
    plt.xlabel("epochs")
    plt.ylabel("accuracies (%)")
    plt.legend(["train", "test"])
    plt.pause(0.01)
    plt.ioff()
    plt.show()

    # SAVE THE MODEL TO DISK
    net.save("mlp2.npz")


def multinomial_logreg_inference(X, W, b):
    logits = X @ W + b.T
    # softmax
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return probs


def multinomial_logreg_train(X, Y, lr=0.005, steps=10000):
    m, n = X.shape
    k = (Y.max() + 1).astype('int')  # number of classes
    W = np.zeros((n, k))
    b = np.zeros(k)
    accuracies = []
    # Build the one hot vectors H
    H = np.zeros((m, k))
    H[np.arange(m), Y.astype('int')] = 1
    for step in range(steps):
        P = multinomial_logreg_inference(X, W, b)
        if step % 100 == 0:
            Yhat = (P > 0.5)
            Yhat = np.argmax(Yhat, 1)
            accuracy = (Yhat == Y).mean()
            accuracies.append(100 * accuracy)
        grad_W = (X.T @ (P - H)) / m
        grad_b = (P - H).mean(0)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b, accuracies


# building the Logistic Regression Classifier
def multinomial_logreg(Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation):
    print("Pls Wait")
    w, b, acc = multinomial_logreg_train(Xtrain, Ytrain)
    predictions = multinomial_logreg_inference(Xtrain, w, b)
    predictions = np.argmax(predictions, 1)
    accuracy_train = (predictions == Ytrain).mean()
    # show_confusion_matrix(Ytrain, predictions)
    # plt.show()
    plt.plot(acc)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy %")
    plt.show()

    print("Pls Wait")
    w, b, acc = multinomial_logreg_train(Xtest, Ytest)
    predictions = multinomial_logreg_inference(Xtest, w, b)
    predictions = np.argmax(predictions, 1)
    accuracy_test = (predictions == Ytest).mean()
    show_confusion_matrix(Ytest, predictions)
    plt.show()
    print("Pls Wait")
    w, b, acc = multinomial_logreg_train(Xvalidation, Yvalidation)
    predictions = multinomial_logreg_inference(Xvalidation, w, b)
    predictions = np.argmax(predictions, 1)
    accuracy_validation = (predictions == Yvalidation).mean()
    show_confusion_matrix(Yvalidation, predictions)
    plt.show()

    print("----------------------")
    print("Accuracies for Logistic Regression Classifier:")
    print("Training accuracy:", accuracy_train * 100)
    print("Testing accuracy:", accuracy_test * 100)
    print("Validation accuracy:", accuracy_validation * 100)


def svm_train(X, Y, lambda_=0.0, lr=1e-3, steps=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for step in range(steps):
        z = X @ w + b
        hinge_diff = ~Y * (z < 1) + (1 - Y) * (z > -1)
        grad_w = (hinge_diff @ X) / m + lambda_ * w
        grad_b = hinge_diff.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def one_vs_rest_train(X, Y, lambda_=0.0, lr=1e-3, steps=1000):
    k = (Y.max() + 1).astype('int')
    W = np.zeros((X.shape[1], k))
    b = np.zeros(k)
    for c in range(k):
        Ybin = (Y == c)
        wbin, bbin = svm_train(X, Ybin, lambda_, lr, steps)
        W[:, c] = wbin
        b[c] = bbin
    return W, b


def one_vs_rest_inference(X, W, b):
    scores = X @ W + b.T
    labels = scores.argmax(1)
    return labels


def one_vs_rest(Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation):
    print("Pls Wait")
    # w, b = one_vs_rest_train(Xtrain, Ytrain)
    # predictions = one_vs_rest_inference(Xtrain, w, b)
    # accuracy_train = (predictions == Ytrain).mean()
    # show_confusion_matrix(Ytrain, predictions)
    # plt.show()

    print("Pls Wait")
    w, b = one_vs_rest_train(Xtest, Ytest)
    predictions = one_vs_rest_inference(Xtest, w, b)
    print(predictions.shape)
    print(Ytest.shape)
    accuracy_test = (predictions == Ytest).mean()
    show_confusion_matrix(Ytest, predictions)
    plt.show()

    print("Pls Wait")
    w, b = one_vs_rest_train(Xvalidation, Yvalidation)
    predictions = one_vs_rest_inference(Xvalidation, w, b)
    accuracy_validation = (predictions == Yvalidation).mean()
    show_confusion_matrix(Yvalidation, predictions)
    plt.show()

    print("----------------------")
    print("Accuracies for One Vs Rest SVM:")
    # print("Training accuracy:", accuracy_train * 100)
    print("Testing accuracy:", accuracy_test * 100)
    print("Validation accuracy:", accuracy_validation * 100)


def one_vs_one_train(X, Y, lambda_=0.0, lr=0.05, steps=10000):
    k = (Y.max() + 1).astype('int')
    m, n = X.shape
    W = np.zeros((n, k * (k - 1) // 2))
    b = np.zeros(k * (k - 1) // 2)
    j = 0
    # For each pair of classes ...
    for pos in range(k):
        for neg in range(pos + 1, k):
            # Build a training subset
            subset = (np.logical_or(Y == pos, Y == neg)).nonzero()[0]
            Xbin = X[subset, :]
            Ybin = (Y[subset] == pos)
            # Train the classifier
            Wbin, bbin = svm_train(Xbin, Ybin, lambda_, lr, steps)
            W[:, j] = Wbin
            b[j] = bbin
            j += 1
    return W, b


def one_vs_one_inference(X, W, b):
    # 1) recover the number of classes from s = 1 + 2 + ... + k
    m = X.shape[0]
    s = b.size
    k = int(1 + np.sqrt(1 + 8 * s)) // 2
    votes = np.zeros((m, k))
    scores = X @ W + b.T
    bin_labels = (scores > 0)
    # For each pair of classes ...
    j = 0
    for pos in range(k):
        for neg in range(pos + 1, k):
            votes[:, pos] += bin_labels[:, j]
            votes[:, neg] += (1 - bin_labels[:, j])
            j += 1
    labels = np.argmax(votes, 1)
    return labels


def one_vs_one(Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation):
    print("Pls Wait")
    # w, b = one_vs_one_train(Xtrain, Ytrain)
    # predictions = one_vs_one_inference(Xtrain, w, b)
    # accuracy_train = (predictions == Ytrain).mean()
    # show_confusion_matrix(Ytrain, predictions)
    # plt.show()

    print("Pls Wait")
    w, b = one_vs_one_train(Xtest, Ytest)
    predictions = one_vs_one_inference(Xtest, w, b)
    print(predictions.shape)
    print(Ytest.shape)
    accuracy_test = (predictions == Ytest).mean()
    show_confusion_matrix(Ytest, predictions)
    plt.show()

    print("Pls Wait")
    w, b = one_vs_one_train(Xvalidation, Yvalidation)
    predictions = one_vs_one_inference(Xvalidation, w, b)
    accuracy_validation = (predictions == Yvalidation).mean()
    show_confusion_matrix(Yvalidation, predictions)
    plt.show()

    print("----------------------")
    print("Accuracies for One Vs One SVM:")
    # print("Training accuracy:", accuracy_train * 100)
    print("Testing accuracy:", accuracy_test * 100)
    print("Validation accuracy:", accuracy_validation * 100)


# DISPLAY THE CONFUSION MATRIX
def show_confusion_matrix(Y, predictions):
    classes = (Y.max() + 1).astype('int')
    cm = np.empty((classes, classes))
    for klass in range(classes):
        sel = (Y == klass).nonzero()
        counts = np.bincount(predictions[sel], minlength=classes)
        cm[klass, :] = 100 * counts / max(1, counts.sum())
    plt.figure(3)
    plt.clf()
    plt.imshow(cm, vmin=0, vmax=100, cmap=plt.cm.Blues)
    for i in range(classes):
        for j in range(classes):
            txt = "{:.1f}".format(cm[i, j], ha="center", va="center")
            col = ("black" if cm[i, j] < 75 else "white")
            plt.text(j - 0.25, i, txt, color=col)
    plt.title("Confusion matrix")


def short_description():
    # Load the training, Validation, and Test data
    data = np.loadtxt("train_short.txt.gz")
    Xtrain = data[:, :-1]
    Ytrain = data[:, -1]

    data = np.loadtxt("test_short.txt.gz")
    Xtest = data[:, :-1]
    Ytest = data[:, -1]

    data = np.loadtxt("validation_short.txt.gz")
    Xvalidation = data[:, :-1]
    Yvalidation = data[:, -1]

    return Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation


def detailed_description():
    # Load the training, Validation, and Test data
    data = np.loadtxt("train.txt.gz")
    Xtrain = data[:, :-1]
    Ytrain = data[:, -1]

    data = np.loadtxt("test.txt.gz")
    Xtest = data[:, :-1]
    Ytest = data[:, -1]

    data = np.loadtxt("validation.txt.gz")
    Xvalidation = data[:, :-1]
    Yvalidation = data[:, -1]

    return Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation


def main():
    # Load the list of classes
    words = open("classes.txt").read().split()
    print(words)

    # to read the detailed description, as in the whole document
    Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation = detailed_description()

    # to read only the short description, as in only the first line
    # Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation = short_description()

    print(Xtrain.shape, Ytrain.shape)
    print(Xtest.shape, Ytest.shape)
    print(Xvalidation.shape, Yvalidation.shape)

    # MEAN/VARIANCE NORMALIZATION
    # Xtrain, Xtest, Xvalidation = meanvar_normalization(Xtrain, Xtest, Xvalidation)

    # Xtrain = l2_normalization(Xtrain)
    # Xtest = l2_normalization(Xtest)
    # Xvalidation = l2_normalization(Xvalidation)

    # naive_bayes(Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation)
    # multi_layer_perceptron(Xtrain, Ytrain, Xtest, Ytest)
    # gaussian_naive_bayes(Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation)
    multinomial_logreg(Xtrain, Ytrain, Xtest, Ytest, Xvalidation, Yvalidation)


if __name__ == "__main__":
    main()
