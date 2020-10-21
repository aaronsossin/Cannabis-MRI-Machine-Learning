from matplotlib import pyplot as plt
import numpy as np
class plot_participant_data:
    def __init__(self, participant_data, cats=[]):
        self.participant_data = participant_data
        self.cats = cats

    def plot_cats(self):
        fig = plt.figure()
        for x in range(len(self.cats)):
            vals = list(self.participant_data[self.cats[x]])
            xs = [x] * len(vals)
            plt.scatter(xs, vals)
        plt.title("Visualization of Participant Data")
        plt.xlabel("Category")
        plt.ylabel("Score")
        plt.xticks(np.arange(0, len(self.cats)), self.cats)
        plt.show()


