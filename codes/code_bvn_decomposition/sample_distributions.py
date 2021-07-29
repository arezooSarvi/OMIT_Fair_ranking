import numpy as np
import matplotlib.pyplot as plt


def sample_normal_distribution(sample_size=10, std=1, mean=0):
    return np.random.normal(loc=mean, scale=std, size=sample_size)


def sample_powerlaw_distribution(sample_size=10, alpha=1):
    exp = -1 / alpha
    return np.array([u ** exp for u in np.random.random(sample_size)])


def sample_uniform_distribution(sample_size=10):
    return np.random.uniform(low=0, high=1, size=sample_size)


def plot_samples():
    for i in range(3):
        plt.figure()
        predictedY = sample_powerlaw_distribution(200)
        predictedY.sort()
        std = predictedY.std()
        mean = predictedY.mean()
        UnlabelledY = [abs(i - mean) > 2 * std for i in predictedY]

        plt.scatter(
            predictedY, np.zeros_like(predictedY), c=UnlabelledY, cmap="hot_r", vmin=-2
        )

        plt.yticks([])
        plt.title("Sample 200 from powerlaw")
        plt.show()

        plt.figure()
        predictedY1 = predictedY[: 200 - 10]
        std1 = predictedY1.std()
        mean1 = predictedY1.mean()
        UnlabelledY1 = [abs(i - mean1) > 2 * std1 for i in predictedY1]

        plt.scatter(
            predictedY1,
            np.zeros_like(predictedY1),
            c=UnlabelledY1,
            cmap="hot_r",
            vmin=-2,
        )

        plt.yticks([])
        plt.title("Removed 10 largest elements")
        plt.show()

        plt.figure()
        predictedY2 = predictedY[200 - 10 :]
        std2 = predictedY2.std()
        mean2 = predictedY2.mean()
        UnlabelledY2 = [abs(i - mean2) > 2 * std2 for i in predictedY2]

        plt.scatter(
            predictedY2,
            np.zeros_like(predictedY2),
            c=UnlabelledY2,
            cmap="hot_r",
            vmin=-2,
        )

        plt.yticks([])
        plt.title("The 10 largest elements")
        plt.show()
