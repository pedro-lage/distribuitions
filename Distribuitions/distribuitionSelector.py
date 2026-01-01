import matplotlib.pyplot as plt
import numpy as np
import colorsys

class DistribuitionSelector():
    def __init__(self):
        self.distribuitions = []
        self.fit_to = []

    def getColors(self):
        hueRange = np.linspace(0,1,len(self.distribuitions)+1)
        colors = [colorsys.hsv_to_rgb(hue,0.8,0.8) for hue in hueRange]
        return colors

    def appendDistribuition(self,distribuition):
        self.distribuitions.append(distribuition)

    def fit(self,xs,method='mvs'):
        self.fit_to = xs
        for distribuition in self.distribuitions:
            distribuition.fit(xs,method=method)

    def fitTest(self):
        return [distribuition.fitTest() for distribuition in self.distribuitions]
    
    def describePlot(self):
        colors = self.getColors()
        fig, axs = plt.subplots(ncols=3, figsize=(12,4))
        for i,distribuition in enumerate(self.distribuitions):
            distribuition.plot_c = colors[i]
            if i != 0:
                distribuition.plotHist=False
            distribuition.describe_plot(show=False, fig=fig, axs=axs)
        legend = [type(distribuition).__name__ for distribuition in self.distribuitions]
        
        plt.legend(legend)
        plt.show()

    def getParams(self):
        return [[distribuition.position,distribuition.scale,distribuition.shape] for distribuition in self.distribuitions]


