# General imports
import matplotlib.pyplot as plt
import copy
import pandas as pd
from abc import ABC
import numpy as np
import warnings

# Lmons imports
import lmoments3 as lm
from lmoments3 import stats as lmstats
from lmoments3 import distr as lmdistr

# Scipy imports
from scipy import stats

# Local imports
from Distribuitions.distr_sources import DistribuitionSource


class Distribuition(ABC):
    # CONSTRUCTOR
    def __init__(self, distribuition, lmoments_distr_name, distribuition_source=DistribuitionSource.LMOMENTS3 ):
        
        # "Default distribuition soure is LMOMENTS3"
        
        self.distribuition = distribuition 
        self.lmoments_distr_name = lmoments_distr_name
        self._distribuition_source = distribuition_source
        self.scipy_params = {}
        self.position = 0
        self.scale = 0
        self.shape = 0
        self.plot_min = 0
        self.plot_max = 0
        self.plot_spacing = 0
        self.plot_multiplier = 1.5
        self.plot_points = 100
        self.fit_to = []
        self.plot_bins = 10
        self.show_hist = False
        self.positiveOnly = False
        self.negativeOnly = False
        self.plot_c = '#5e1919'

        self.fit_kwargs = {}

    # PROPERTIES (PRIVATE)
    def set_plot_multiplier(self,multiplier):
        self.multiplier = multiplier
        self.update_plot_properties()

    def set_position(self,position):
        self.position = position
        self.update_plot_properties()

    def set_scale(self,scale):
        self.scale = scale
        self.update_plot_properties()

    def set_shape(self,shape):
        self.shape = shape
        self.update_plot_properties()

    def update_plot_properties(self):
        
        self.plot_min = np.min(self.fit_to)
        if self.plot_min > 0:
            self.plot_min = self.plot_min/self.plot_multiplier
        else:
            if self.positiveOnly:
                self.plot_min = 0
            else:
                self.plot_min = self.plot_min*self.plot_multiplier
            
    
        self.plot_max = np.max(self.fit_to)
        if self.plot_max > 0:
            self.plot_max = self.plot_max*self.plot_multiplier
        else:
            if self.negativeOnly:
                self.plot_max = 0
            else:
                self.plot_max = self.plot_max/self.plot_multiplier


        self.plot_spacing = ((self.plot_max - self.plot_min) / self.plot_points)

    def get_hist_c(self):
        return '#6daba5'

    def get_shape_param_list(self):
        return(['a','c','skew'])
    
    def get_scale_param_list(self):
        return(['scale'])
    
    def get_position_param_list(self):
        return(['loc'])

    # PUBLIC METHODS
    def sample_lmoms(self):
        return lm.lmom_ratios(self.fit_to,nmom=4)

    def fit(self,xs,method='mvs'):
        self.fit_to = xs
        if len(xs) > 0:
            self.show_hist = True

        if method == 'mom':
            self.fit_mom()
        elif method == 'mml':
            self.fit_mml()
        elif method == 'mvs':
            self.fit_mvs()
        else:
            raise NotImplementedError 
        
    def sumsquare_error(self):
        sorted_data = copy.deepcopy(self.fit_to)
        sorted_data = sorted_data.tolist()
        sorted_data.sort()
        sorted_data = np.array(sorted_data)
        emp_probs = np.array(list(range(len(sorted_data))))/len(sorted_data)
        return np.sum((self.cdf_list(sorted_data)-emp_probs)**2)/len(emp_probs)
        
    def fitTest_ChiSquare(self):
        raise NotImplementedError 

    def fitTest_KolmogorovSmirnov(self):
        return stats.kstest(rvs=self.fit_to,cdf=self.cdf)

    def fitTest_AndersonDarling(self):
        raise NotImplementedError 

    def fitTest_Filliben(self):
        raise NotImplementedError 

    # TODO: Implementar AIC e BIC para distribuições do scipy
    def AIC(self):
        if self._distribuition_source == DistribuitionSource.LMOMENTS3:
            return lmstats.AICc(data = self.fit_to, distr_name=self.lmoments_distr_name, distr_paras=self.scipy_params)
        else:
            raise NotImplementedError

    # TODO: Implementar AIC e BIC para distribuições do scipy
    def BIC(self):
        if self._distribuition_source == DistribuitionSource.LMOMENTS3:
            return lmstats.BIC(data = self.fit_to, distr_name=self.lmoments_distr_name, distr_paras=self.scipy_params)
        else:
            raise NotImplementedError
        
    # TODO: Implementar AIC e BIC para distribuições do scipy
    def fitTest(self):
        if self._distribuition_source == DistribuitionSource.LMOMENTS3:
            return {'KS_pvalue':self.fitTest_KolmogorovSmirnov().pvalue, 'SSE': self.sumsquare_error(), 'AIC':self.AIC(), 'BIC':self.BIC()}
        else:
            warnings.warn("AIC and BIC not yet implemented outside LEMOMENTS3 distribuitions.")
            return {'KS_pvalue':self.fitTest_KolmogorovSmirnov().pvalue, 'SSE': self.sumsquare_error(), 'AIC':-np.inf, 'BIC':-np.inf}

    def fit_mom(self):
        raise NotImplementedError 
        params = self.distribuition.fit(self.fit_to,method='MM')
        self._set_dist_params(params)

    def fit_mvs(self):

        params = self.distribuition.fit(self.fit_to, 
                                        **self.fit_kwargs)

        self._set_dist_params(params, method = 'mvs')

    # TODO: Implementar fit mml para distribuições do scipy
    def fit_mml(self):

        if not self._distribuition_source == DistribuitionSource.LMOMENTS3:
            raise NotImplementedError("MML fitting method is only available for LMOMENTS3 distributions.")
        
        try:
            params = self.distribuition.lmom_fit(self.fit_to)
        except:
            return
        self._set_dist_params(params, method = 'mml')

    def pdf(self,x):
        if self._distribuition_source == DistribuitionSource.LMOMENTS3:
            return self.distribuition.pdf(x=x,**self.scipy_params)
        elif self._distribuition_source == DistribuitionSource.SCIPY:
            return self.distribuition.pdf(x,*self.scipy_params)
        else:
            raise NotImplementedError("Distribuition source not implemented.")

    def cdf(self,x):
        if self._distribuition_source == DistribuitionSource.LMOMENTS3:
            return self.distribuition.cdf(x=x,**self.scipy_params)
        elif self._distribuition_source == DistribuitionSource.SCIPY:
            return self.distribuition.cdf(x,*self.scipy_params)
        else:
            raise NotImplementedError("Distribuition source not implemented.")
        
    def ppf(self,q):
        if self._distribuition_source == DistribuitionSource.LMOMENTS3:
            return self.distribuition.ppf(q=q,**self.scipy_params)
        elif self._distribuition_source == DistribuitionSource.SCIPY:
            return self.distribuition.ppf(q,*self.scipy_params)
        else:
            raise NotImplementedError("Distribuition source not implemented.")
        
    def plot_pdf(self,ax,show=False):
        if self.show_hist:
            ax.hist(self.fit_to,bins=self.plot_bins,density=True,color=self.get_hist_c())
        xs = self._arange_plot()
        ax.plot(xs,self.pdf_list(xs),color=self.plot_c)

        ax.set_title("Função de Densidade de Probabilidade")
        ax.set_xlabel("Variável")
        ax.set_ylabel("Densidade de Probabilidade")

        if show:
            plt.show()
    
    def plot_cdf(self,ax,show=False):
        if self.show_hist:
            self.__cum_hist(self.fit_to,ax,self.plot_bins)
        xs = self._arange_plot()
        ax.plot(xs,self.cdf_list(xs),color=self.plot_c)

        ax.set_title("Função de Probabilidade Acumulada")
        ax.set_xlabel("Variável")
        ax.set_ylabel("Probabilidade de Não Excedência")

        if show:
            plt.show()  
    
    def plot_ppf(self,ax,show=False):
        xs = self._arange_plot()
        ax.plot(self.cdf_list(xs),xs,color=self.plot_c)

        ax.set_title("Função de Quantis")
        ax.set_xlabel("Probabilidade de Não Excedência")
        ax.set_ylabel("Variável")

        if show:
            plt.show()

    def describe_plot(self,show=True, fig="", axs=""):
        if show:
            fig, axs = plt.subplots(ncols=3, figsize=(14,4))
        self.plot_pdf(axs[0])
        self.plot_cdf(axs[1])
        self.plot_ppf(axs[2])
        if show:
            plt.show()

    def pdf_list(self,xs):
        return [self.pdf(x) for x in xs]
    
    def cdf_list(self,xs):
        return [self.cdf(x) for x in xs]
    
    def ppf_list(self,ps):
        return [self.ppf(p) for p in ps]
    
    def generate(self):
        return self.ppf(np.random.rand())
    
    def generate_list(self,len):
        return [self.ppf(np.random.rand()) for x in range(len)]
    
    # PRIVATE METHODS
    def _arange_plot(self):
        return np.arange(start=self.plot_min,stop=self.plot_max,step=self.plot_spacing)
    
    def _set_dist_params(self,params, method = 'mml'):

        if method == 'mml':
            if self._distribuition_source == DistribuitionSource.LMOMENTS3:
                set_shape = False
                for shape_param in self.get_shape_param_list():
                    if shape_param in params:
                        self.set_shape(params[shape_param])
                        shape = params[shape_param]
                        set_shape = True
                set_scale = False
                for scale_param in self.get_scale_param_list():
                    if scale_param in params:
                        self.set_scale(params[scale_param])
                        scale = params[scale_param]
                        set_scale = True
                set_position = False
                for position_param in self.get_position_param_list():
                    if position_param in params:
                        self.set_position(params[position_param])
                        position = params[position_param]
                        set_position = True
                
                self.scipy_params = {}
                if set_shape:
                    if self.lmoments_distr_name == 'gam':
                        self.scipy_params["a"] = shape
                    elif self.lmoments_distr_name == 'gev':
                        self.scipy_params["c"] = shape
                    elif self.lmoments_distr_name == 'pe3':
                        self.scipy_params["skew"] = shape
                    elif self.lmoments_distr_name == 'wei':
                        self.scipy_params["c"] = shape

                if set_scale:
                    self.scipy_params["scale"] = scale
                if set_position:
                    self.scipy_params["loc"] = position
            
            else:
                raise NotImplementedError("MML fitting method is only available for LMOMENTS3 distributions.")
                   
        elif method == 'mvs':

            if self._distribuition_source == DistribuitionSource.SCIPY:
                self.scipy_params = params

            elif self._distribuition_source == DistribuitionSource.LMOMENTS3:
                if len(params) == 1:
                    self.set_position(params[0])
                    self.scipy_params = {"loc": params[0]}

                elif len(params) == 2:
                    self.set_position(params[0])
                    self.set_scale(params[1])
                    self.scipy_params = {"loc": params[0], "scale": params[1]}
                    if self.lmoments_distr_name == 'exp':
                        self.scipy_params = {"loc": params[0], "scale": params[1]}

                else:
                    self.set_position(params[2])
                    self.set_scale(params[1])
                    self.set_shape(params[0])

                    self.scipy_params = {"loc": params[2], "scale": params[1]}
                    if self.lmoments_distr_name in ['gam','wei']:
                        self.scipy_params = {"loc": params[1], "scale": params[2]}

                    if self.lmoments_distr_name == 'gam':
                        self.scipy_params["a"] = params[0]
                    elif self.lmoments_distr_name == 'gev':
                        self.scipy_params["c"] = params[0]
                    elif self.lmoments_distr_name == 'pe3':
                        self.scipy_params["skew"] = params[0]
                    elif self.lmoments_distr_name == 'wei':
                        self.scipy_params["c"] = params[0]
        
            else:
                raise NotImplementedError("Distribuition source not implemented.")  
            
            self.update_plot_properties()

    # PROTECTED METHODS
    def __cum_hist(self,fit_to,ax,bins):
        res = stats.cumfreq(fit_to,
                    numbins=bins)

        xs = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size,
                                res.cumcount.size)

        ys = [y/len(fit_to) for y in res.cumcount]

        colors = [self.get_hist_c() for y in res.cumcount]

        ax.bar(xs, ys, color=colors, width=xs[1]-xs[0])

