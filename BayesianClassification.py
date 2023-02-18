import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Handle data
data_train1 = pd.read_csv('parta_Train1.csv', header=None)
data_test1 = pd.read_csv('parta_Test1.csv', header=None)
# ///////////////////////
data_train2 = pd.read_csv('parta_Train2.csv', header=None)
data_test2 = pd.read_csv('parta_Test2.csv', header=None)


# Calculate Gaussian Probability Density Function
def gaussianPdf(x, mean, stdev):
    power = mean.shape[0]
    stdevDet = np.linalg.det(stdev)
    stdevInv = np.linalg.inv(stdev)

    Denominator = np.sqrt((2 * np.pi) ** power * stdevDet)
    fc = np.einsum('...k,kl,...l->...', x - mean, stdevInv, x - mean)  # Einstein summation convention
    return np.exp(-fc / 2) / Denominator


def predict(x, mean, stdev):
    power = mean.shape[0]
    stdevDet = np.linalg.det(stdev)
    stdevInv = np.linalg.inv(stdev)

    Denominator = np.sqrt((2 * np.pi) ** power * stdevDet)
    fc = np.einsum('...k,kl,...l->...', x - mean, stdevInv, x - mean)
    return np.exp(-fc / 2) / Denominator


def Accuracy(True_value, predict_0, predict_1):
    true_itm = 0
    bound = 0
    for i in range(len(True_value)):
        if (predict_0[i] > predict_1[i]) and (True_value[i] == 0):
            true_itm += 1
        elif (predict_0[i] < predict_1[i]) and True_value[i] == 1:
            true_itm += 1
        elif predict_0[i] == predict_1[i]:
            bound += 1

    return (true_itm * 100) / len(True_value)


def plot_fit(data, mean0, mean1, stdev, Title, fig_show=False):
    x = data[0]
    y = data[1]

    mu0 = np.array(mean0)
    mu1 = np.array(mean1)

    grouped_data = data.groupby(data[2])

    p0 = grouped_data.get_group(0).shape[0] / data.shape[0]
    p1 = grouped_data.get_group(1).shape[0] / data.shape[0]
    stdevInv = np.linalg.inv(stdev)

    a = np.dot(stdevInv, (mu0 - mu1))
    b = (((np.dot(np.dot(mu0.T, stdevInv), mu0)) / 2) - ((np.dot(np.dot(mu1.T, stdevInv), mu1)) / 2)) + np.log(
        p0 / p1) - 5

    x_d = np.array([-1, 0, 5])

    y_d = -(b + x_d * a[0]) / a[1]

    fig = plt.figure(figsize=(5, 3.75))
    ax2d = fig.add_subplot(111)

    ax2d.scatter(grouped_data.get_group(0)[0], grouped_data.get_group(0)[1], color='b', label='class 0',
                 cmap=plt.cm.binary, zorder=2)
    ax2d.scatter(grouped_data.get_group(1)[0], grouped_data.get_group(1)[1], color='r', label='class 0',
                 cmap=plt.cm.binary, zorder=2)

    ax2d.plot(x_d, y_d, color='b')
    plt.title = Title
    if fig_show:
        plt.figure()
    plt.show()


# -----------------------------Train---------------------------
def main(dataTrain, dataTest):
    classData = dataTrain.groupby(dataTrain[2])  # group by data by class 0,1

    mean0 = classData.get_group(0)[[0, 1]].mean()  # mean of class 0
    covariance0 = classData.get_group(0)[[0, 1]].cov()  # covariance of class 0

    mean1 = classData.get_group(1)[[0, 1]].mean()  # mean of class 1
    covariance1 = classData.get_group(1)[[0, 1]].cov()  # covariance of class 1

    meanAll = dataTrain[[0, 1]].mean()  # mean of all class
    covarianceAll = dataTrain[[0, 1]].cov()  # covariance of all class

    # create a grid of x & y for plot pdf
    n = 60
    x = np.linspace(-5, 10, n)
    y = np.linspace(-5, 11, n)
    x, y = np.meshgrid(x, y)

    pos = np.empty(x.shape + (2,))  # add 2 dimension for pack x,y
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    z_0 = gaussianPdf(pos, np.array(mean0), np.array(covariance1))  # z dimension for class 0

    z_1 = gaussianPdf(pos, np.array(mean1), np.array(covariance1))  # z dimension for class 1

    fig = plt.figure(figsize=(12, 6))  # plot gaussian class
    ax3d = fig.gca(projection='3d')
    ax3d.plot_surface(x, y, z_0, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax3d.plot_surface(x, y, z_1, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax3d.view_init(27, -21)
    plt.grid()
    # plt.show()

    z_train_0 = predict(dataTrain[[0, 1]], mean0, covariance0)  # predict for Train class0
    z_train_1 = predict(dataTrain[[0, 1]], mean1, covariance0)  # predict for Train class 1

    print("Train Accuracy:", Accuracy(dataTrain[2], z_train_0, z_train_1))  # print Accuracy of train data
    print("covariance class 0:\n", covariance0)
    print("covariance class 1:\n", covariance1)
    print("covariance all class:\n", covarianceAll)

    plot_fit(dataTrain, mean0, mean1, covarianceAll, 'Train')

    # Test

    z_test_0 = predict(dataTest[[0, 1]], mean0, covariance0)  # get predict of class0 test
    z_test_1 = predict(dataTest[[0, 1]], mean1, covariance0)  # get predict of class0 test

    print("Test Accuracy:", Accuracy(dataTest[2], z_test_0, z_test_1))  # print Accuracy of test data


print("data 1")
main(data_train1, data_test1)
print("\n\ndata 2")
main(data_train2, data_test2)
