import matplotlib.pyplot as plt
import os
import settings

class livePlot():
    def __init__(self):
        self.data = None
        self.eps_data = None
        self.score_data = None
        self.loss_data = None
        self.epochs = 0

    def update_plot(self, stats):
        self.data = stats["AvgReturns"]
        self.score_data = stats["HighScore"]
        self.loss_data = stats["Loss"]
        self.eps_data = stats["EpsilonCheckpoint"]

        self.epochs = len(self.data)

        self.ax.clear()
        self.ax.set_xlim(0, self.epochs)

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Episodes * 10")
        self.ax.set_title("Training Stats")

        self.ax.plot(self.data, 'o-', label='AvgReturns[-100:]')
        self.ax.plot(self.eps_data, 'r-', label='Epsilon')
        self.ax.plot(self.loss_data, 'b-', label='Loss')
        self.ax.plot(self.score_data, 'g-', label='High Score')
        self.ax.legend(loc='upper left')

        if not os.path.exists('plots'):
            os.makedirs('plots')

        self.fig.savefig(f'plots/{settings.GRAPH_NAME}.png')
        