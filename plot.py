import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

if __name__ == '__main__':

    with open('rewards_losses.pkl', 'rb') as handle:
        rewards, losses = pkl.load(handle)

    plt.figure()
    plt.plot(range(len(rewards)), rewards)
    plt.plot(range(len(losses)), losses)
    plt.show()
