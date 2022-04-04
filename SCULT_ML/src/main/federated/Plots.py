from typing import Any

import numpy as np
import matplotlib.pyplot as plt


class Plot:

    def __init__(self, rounds):
        super()
        self._rounds = rounds
        self._x1 = np.arange(0, self._rounds, 1, None)
        self._x2 = np.arange(0, self._rounds, 1, None)
        self.y1 = np.zeros(self._rounds)
        self.y2 = np.zeros(self._rounds)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def set_accuracy(self, round, val):
        self.y1[round] = val

    def set_loss(self, round, val):
        self.y2[round] = val

    def plot_scatter(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('Validation Accuracy and Loss')

        ax1.plot(self._x1, self.y1, 'o-')
        ax1.set_title('model accuracy')
        ax1.set_ylabel('accuracy')
        ax1.legend(['train', 'test'], loc='upper left')

        ax2.plot(self._x2, self.y2, '.-')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'test'], loc='upper left')
        plt.show()

    @property
    def rounds(self):
        return self._rounds

    @rounds.setter
    def rounds(self, rounds):
        self._rounds = rounds
