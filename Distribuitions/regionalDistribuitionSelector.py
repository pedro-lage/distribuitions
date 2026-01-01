import matplotlib.pyplot as plt
import numpy as np
import colorsys
from scipy import stats
from copy import copy

from distribuitionSelector import DistribuitionSelector


class RegionalDistribuitionSelector():
    def __init__(self):
        self.distribuitions = []
        self.selectors = []
        self.fit_to = []
        self.vogelPlotLength = 10000
        self.method = 'mvs'
        self.significance = 0.05

    def getColors(self):
        hueRange = np.linspace(0,1,len(self.distribuitions)+1)
        colors = [colorsys.hsv_to_rgb(hue,0.8,0.8) for hue in hueRange]
        return colors

    def appendDistribuition(self,distribuition):
        self.distribuitions.append(distribuition)

    def fitData(self,xss):
        self.selectors = []
        self.fit_to = xss

        for xs in xss:
            distribuitionselector = DistribuitionSelector()
            for distribuition in self.distribuitions:
                distribuitionselector.appendDistribuition(copy.deepcopy(distribuition))
            distribuitionselector.fit(xs,method=self.method)
            self.selectors.append(distribuitionselector)

    def vogelPlot(self):
        fig, axs = plt.subplots(ncols=1, figsize=(5,5))
        skews = []
        kurts = []
        for xs in self.fit_to:
            skews.append(stats.skew(xs))
            kurts.append(stats.kurtosis(xs))
        axs.scatter([skews],[kurts],c='black',s=5)

        colors = self.getColors()
        for selector in self.selectors:
            for distribuition,c in zip(selector.distribuitions,colors):
                try:
                    sample = distribuition.generate_list(self.vogelPlotLength)
                    skew = stats.skew(sample)
                    kurt = stats.kurtosis(sample)
                    axs.scatter([skew],[kurt],c=[c],s=5)
                except:
                    continue

        plt.xlabel('skewness')
        plt.ylabel('kurtosis')
        
        legend = ['Data']
        [legend.append(type(distribuition).__name__) for distribuition in self.distribuitions]
        plt.legend(legend)
        plt.show()


#{'KS_pvalue':self.fitTest_KolmogorovSmirnov().pvalue, 'SSE': self.sumsquare_error(), 'AIC':self.AIC(), 'BIC':self.BIC()}
    def goodnessOfFitResults(self):
        fig, axs = plt.subplots(nrows=4, figsize=(10,20))
        distNames = [type(distribuition).__name__ for distribuition in self.distribuitions]
        distResults = {distName:{'accepted':0,'rejected':0, 'acceptanceRate': 0, 'meanPvalue':0, 'pvalues':[], 'SSEs':[], 'AICs':[], 'BICs':[]} for distName in distNames}
        for selector in self.selectors:
            fitTestResults = selector.fitTest()
            for distribuition, fitTestResult in zip(selector.distribuitions,fitTestResults):
                distName = type(distribuition).__name__
                distResults[distName]['pvalues'].append(fitTestResult['KS_pvalue'])
                distResults[distName]['SSEs'].append(fitTestResult['SSE'])
                distResults[distName]['AICs'].append(fitTestResult['AIC'])
                distResults[distName]['BICs'].append(fitTestResult['BIC'])
                if fitTestResult['KS_pvalue'] > self.significance:
                    distResults[distName]['accepted'] = distResults[distName]['accepted'] + 1
                else:
                    distResults[distName]['rejected'] = distResults[distName]['rejected'] + 1
        
        pvalues = []
        SSEs = []
        AICs = []
        BICs = []
        for distribuition in self.distribuitions:
            distName = type(distribuition).__name__
            distResults[distName]['acceptanceRate'] = distResults[distName]['accepted']/(distResults[distName]['accepted']+distResults[distName]['rejected'])
            distResults[distName]['meanPvalue'] = np.nanmean(distResults[distName]['pvalues'])
            pvalues.append(distResults[distName]['pvalues'])
            SSEs.append(distResults[distName]['SSEs'])
            AICs.append(distResults[distName]['AICs'])
            BICs.append(distResults[distName]['BICs'])

        axs[0].boxplot(x=pvalues)
        axs[0].plot([1,len(pvalues)],[self.significance,self.significance],c='red')
        axs[0].set_xticklabels(distNames)
        axs[0].set_ylabel('pvalues')

        axs[1].boxplot(x=SSEs, showfliers=False)
        axs[1].set_xticklabels(distNames)
        axs[1].set_ylabel('SSEs')

        axs[2].boxplot(x=AICs, showfliers=False)
        axs[2].set_xticklabels(distNames)
        axs[2].set_ylabel('AICs')

        axs[3].boxplot(x=BICs, showfliers=False)
        axs[3].set_xticklabels(distNames)
        axs[3].set_ylabel('BICs')
        plt.show()
        
        fig, axs = plt.subplots(nrows=4,ncols=len(self.distribuitions), figsize=(len(self.distribuitions)*3.5,12))
        for i, distribuition in enumerate(self.distribuitions):
            distName = type(distribuition).__name__
            axs[0][i].hist(distResults[distName]['pvalues'],bins=20,range=(0,1))
            axs[0][i].plot([self.significance,self.significance],[0,len(distResults[distName]['pvalues'])/20],c='red')
            axs[0][i].set_xlabel(distName)
            axs[0][i].set_ylabel('pvalues')

            hist_range = self.get_hist_lims(distResults[distName]['SSEs'])
            axs[1][i].hist(distResults[distName]['SSEs'],bins=20,range=hist_range)
            axs[1][i].set_xlabel(distName)
            axs[1][i].set_ylabel('SSEs')

            hist_range = self.get_hist_lims(distResults[distName]['AICs'])
            axs[2][i].hist(distResults[distName]['AICs'],bins=20,range=hist_range)
            axs[2][i].set_xlabel(distName)
            axs[2][i].set_ylabel('AICs')

            hist_range = self.get_hist_lims(distResults[distName]['BICs'])
            axs[3][i].hist(distResults[distName]['BICs'],bins=20,range=hist_range)
            axs[3][i].set_xlabel(distName)
            axs[3][i].set_ylabel('BICs')
        plt.show()

        for distribuition in self.distribuitions:
            distName = type(distribuition).__name__
            del distResults[distName]['pvalues']

        return distResults

    def get_hist_lims(self,xs,quantile = 0.1):
        xs.sort()

        index_min = int(len(xs)*quantile)
        index_max = int(len(xs)*(1-quantile))

        hist_max = xs[index_max]
        hist_min = xs[index_min]

        if np.isinf(hist_max) or np.isnan(hist_max):
            hist_max = np.nanmean(xs) + 3*np.nanstd(xs)
        if np.isinf(hist_min) or np.isnan(hist_min):
            hist_min = np.nanmean(xs) - 3*np.nanstd(xs)

        if np.isnan(hist_max):
            hist_max = 1
        if np.isnan(hist_min):
            hist_min = 0

        if hist_max <= hist_min:
            hist_max = hist_min + 1

        return (hist_min,hist_max)

    def distribuitionParameters(self):
        distNames = [type(distribuition).__name__ for distribuition in self.distribuitions]
        distResults = {distName:{'positions':[], 'scales': [],'shapes':[], 'meanPosition':0, 'meanScale': 0,'meanShape':0} for distName in distNames}
        for selector in self.selectors:
            for distribuition in selector.distribuitions:
                distName = type(distribuition).__name__
                distResults[distName]['positions'].append(float(distribuition.position))
                distResults[distName]['scales'].append(float(distribuition.scale))
                distResults[distName]['shapes'].append(float(distribuition.shape))

        fig, axs = plt.subplots(ncols=len(self.distribuitions), nrows=3, figsize=(len(self.distribuitions)*3.5,9))
        for i, distribuition in enumerate(self.distribuitions):
            distName = type(distribuition).__name__
            axs[0][i].hist(distResults[distName]['positions'],bins=10)
            axs[0][i].set_xlabel(distName + " - posição")
            
            axs[1][i].hist(distResults[distName]['scales'],bins=10)
            axs[1][i].set_xlabel(distName + " - escala")
            
            axs[2][i].hist(distResults[distName]['shapes'],bins=10)
            axs[2][i].set_xlabel(distName + " - forma")
        plt.show()

        for distribuition in self.distribuitions:
            distName = type(distribuition).__name__
            distResults[distName]['meanPosition'] = np.nanmean(distResults[distName]['positions'])
            distResults[distName]['meanScale'] = np.nanmean(distResults[distName]['scales'])
            distResults[distName]['meanShape'] = np.nanmean(distResults[distName]['shapes'])

        for distribuition in self.distribuitions:
            distName = type(distribuition).__name__
            del distResults[distName]['positions']
            del distResults[distName]['scales']
            del distResults[distName]['shapes']

        return distResults
        

# VER Distribuicoes_Validacao para exemplos de uso