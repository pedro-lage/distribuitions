# General imports
import numpy as np
import matplotlib.pyplot as plt

# Lmons imports
import lmoments3 as lm
from lmoments3 import distr as lmdistr

# Scipy imports
from scipy import stats
import scipy.special as special
from scipy.optimize import minimize, root
from scipy.special import gamma

# Local imports
from Distribuitions.distribuition import Distribuition
from Distribuitions.transform_distribuition import TransformDistribuition
from Distribuitions.distr_sources import DistribuitionSource



class Normal(Distribuition):  
    def __init__(self):
        super().__init__(lmdistr.nor,'nor')

class Exponential(Distribuition):
    def __init__(self):
        super().__init__(lmdistr.exp,'exp')
        self.positiveOnly = True

class Gama(Distribuition):
    def __init__(self):
        raise NotImplementedError("Distribuição gama está com erro!")
        super().__init__(lmdistr.gam,'gam')

class GumbelMax(Distribuition):
    def __init__(self):
        super().__init__(lmdistr.gum,'gum')

    def fit_mom(self):
        #raise NotImplementedError 
        media = np.mean(self.fit_to)
        variancia = np.var(self.fit_to)

        scale = np.sqrt((6*variancia)/(np.pi**2))
        loc = media - 0.5772*scale

        self.set_position(loc)
        self.set_scale(scale)
        self.scipy_params["loc"] = loc
        self.scipy_params["scale"] = scale

class GEV(Distribuition):
    def __init__(self):
        super().__init__(lmdistr.gev,'gev')

class WeibullMin(Distribuition):
       
    """IMPORTANTE
    
        1 - Essa distribuição é a Weibull de dois parâmetros, ou seja com o parâmetro de posição igual a zero, o que atribui a distribuição um limite inferior.
            No entanto para o ajuste realizado por Método dos momentos L a distribuição que será ajustada será a Weibull de 3 parâmetros pois 
            a biblioteca útilizada para ajuste por MML não permite setarmos o parâmetro de posição como zero(não que eu saiba).
            
        2 - Até a presente data(15/04/2025) o algorítimo útilizado para cálculo por Método dos Momentos não foi devidamente testado, portanto cuidado ao se útilizar.
    
    """
    def __init__(self):
        super().__init__(lmdistr.wei,'wei')
        self.fit_kwargs = {'floc':0}   #Weibull de mínimos é definida apenas para valores positivos, portanto o parâmetro de posição deve ser fixado em 0]
        
    def fit_mom(self):
        media = np.mean(self.fit_to)
        variancia = np.var(self.fit_to)
        
        # Função para resolver o sistema de equações dos momentos
        def equations(params):
            shape, scale = params
            # Equação da média
            eq1 = scale * gamma(1 + 1/shape) - media
            # Equação da variância
            term1 = gamma(1 + 2/shape)
            term2 = gamma(1 + 1/shape)**2
            eq2 = scale**2 * (term1 - term2) - variancia
            return [eq1, eq2]
        
        # Chutes iniciais razoáveis
        initial_guess = [1.0, media]
        
        # Resolve o sistema de equações não lineares
        solution = root(equations, initial_guess, method='lm')
        
        if not solution.success:
            raise ValueError("Não foi possível convergir para uma solução. Tente outro método de ajuste.")
        
        shape, scale = solution.x
        
        self.set_shape(shape)
        self.set_scale(scale)  
        self.set_position(0)  # O parâmetro de localização é fixado em 0 para a Weibull
        self.scipy_params["loc"] = 0
        self.scipy_params["scale"] = scale
        self.scipy_params["c"] = shape
        self.update_plot_properties()

class WeibullMax(Distribuition):
    
    """
    
        IMPORTANTE

        1 - Essa implementação foi derivada da função de probabilidades acumuladas da distribuição de Weibull para mínimos negativando a variavel 
        aleatória( F(x)-->F(-x) ) em seguida a função densidade de probabilidade foi obtida derivando-se essa equação.
            A respeito dos limites da distribuição, dos limites possíveis para os ajustes dos seus parâmetros e dos limites dos parâmetros
        eu não tenho a menor ideia, por isso, recomendo não útilizar em casos de parâmetros que você tenha dúvidas se estão plausíveis.
    """
    
    def __init__(self):
        super().__init__(lmdistr.wei,'wei')
        self.fit_kwargs = {'floc':0}   #Weibull de mínimos é definida apenas para valores positivos, portanto o parâmetro de posição deve ser fixado em 0]
        
    def pdf(self, x):
        """
        Calcula a PDF da Gumbel de Mínimos usando a fórmula matemática.
        """
        
        scale = -self.scipy_params.get("scale", 1)  # Parâmetro de escala (β)
        shape = self.scipy_params.get("c", 2)  # Parâmetro de forma (α)
        PDF = - (shape / scale) * ((-x/scale) ** (shape - 1)) * np.exp(-((-x/scale) ** shape))
        return PDF

    def cdf(self, x):
        """
        Calcula a CDF da Gumbel de Mínimos usando a fórmula matemática.
        """
        loc = self.scipy_params.get("loc", 0)  # Parâmetro de localização (μ)
        scale = -self.scipy_params.get("scale", 1)  # Parâmetro de escala (β)
        shape = self.scipy_params.get("c", 2)  # Parâmetro de forma (α)
        CDF = 1 - np.exp(-((-x/scale) ** shape))
        return CDF

    def ppf(self, q):
        """
        Calcula a PPF da Gumbel de Mínimos usando a fórmula matemática.
        """
        loc = self.scipy_params.get("loc", 0)  # Parâmetro de localização (μ)
        scale = -self.scipy_params.get("scale", 1)  # Parâmetro de escala (β)
        shape = self.scipy_params.get("c", 2)  # Parâmetro de forma (α)
        PPF = -scale * ((-np.log(q)) ** (1 / shape))
        return PPF

class GumbelMin(Distribuition):
    
    """IMPORTANTE
    
        1 - Essa distribuição foi cálculada pelos métodos de ajustes para a Gumbel de máximos apenas negativando os dados de entrada e por fim
            negativando o parâmetro de posição(loc).
    """
    
    def __init__(self):
        super().__init__(lmdistr.gum, 'gum')
        self.positiveOnly = True  # Gumbel de Mínimos é definida apenas para valores positivos
        
    def fit(self,xs,method='mvs'):
        self.fit_to = -xs       #Negativa os dados de entrada, pois o fit é feito para a Gumbel de Máximos, portanto devemos converter para mínimos
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
        
        self.fit_to = xs     #Retorna os dados de entrada ao normal
        self.update_plot_properties()  #Atualiza as propriedades de plotagem para os dados de entrada não negativos

    def fit_mom(self):
        #raise NotImplementedError 
        media = -1 * (np.mean(self.fit_to)) # Multiplicamos por -1 pois os valores de self.fit_to foram negativados para que fosse possível
                                            # ser feito os ajustes por MML e MVS, mas isso não é necessário aqui
        variancia = np.var(self.fit_to)

        scale = np.sqrt((6*variancia)/(np.pi**2))
        loc = media + 0.5772*scale

        self.set_position(loc)
        self.set_scale(scale)
        self.scipy_params["loc"] = -loc
        self.scipy_params["scale"] = scale

    def pdf(self, x):
        """
        Calcula a PDF da Gumbel de Mínimos usando a fórmula matemática.
        """
        loc = self.scipy_params.get("loc", 0)  # Parâmetro de localização (μ)
        scale = self.scipy_params.get("scale", 1)  # Parâmetro de escala (β)
        loc = -loc      #Inverte o parâmetro de localização, para estar coerente com a Gumbel de Mínimos
        z = (x - loc) / scale
        PDF = (1 / scale) * np.exp(z - np.exp(z))
        return PDF

    def cdf(self, x):
        """
        Calcula a CDF da Gumbel de Mínimos usando a fórmula matemática.
        """
        loc = self.scipy_params.get("loc", 0)
        scale = self.scipy_params.get("scale", 1)
        loc = -loc      #Inverte o parâmetro de localização, para estar coerente com a Gumbel de Mínimos
        z = (x - loc) / scale
        CDF = 1 - np.exp(-np.exp(z))
        return CDF

    def ppf(self, q):
        """
        Calcula a PPF da Gumbel de Mínimos usando a fórmula matemática.
        """
        loc = self.scipy_params.get("loc", 0)
        scale = self.scipy_params.get("scale", 1)
        loc = -loc      #Inverte o parâmetro de localização, para estar coerente com a Gumbel de Mínimos
        PPF = loc + scale * np.log(-np.log(1 - q))
        return PPF

class Frechet(Distribuition):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invweibull.html
    '''
        An inverted Weibull continuous random variable.
        This distribution is also known as the Fréchet distribution or the type II extreme value distribution.
    '''
    def __init__(self):
        super().__init__(stats.invweibull, None, distribuition_source=DistribuitionSource.SCIPY)
        # Na documentação do Scipy, frechet é representada como uma weibull invertida
        self.fit_kwargs = {'floc':0}   # Parâmetro de forma fixo igual a 1 no livro do Mauro (Hidrologia Estatística)

# class LogNormal(Distribuition):
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
#     def __init__(self):
#         super().__init__(stats.lognorm, None, distribuition_source=DistribuitionSource.SCIPY)
#         self.fit_kwargs = {'f0':1}   # Parâmetro de forma fixo igual a 1 no livro do Mauro (Hidrologia Estatística)

class LogNormal(TransformDistribuition, Normal):
    def __init__(self):
        Normal.__init__(self)
        TransformDistribuition.__init__(
            self, 
            transform=np.log, 
            inv_transform=np.exp, 
            jacobian_transform=lambda x: 1/x
        )
        
class GenPareto(Distribuition):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genpareto.html
    def __init__(self):
        super().__init__(stats.genpareto, None, distribuition_source=DistribuitionSource.SCIPY)
        self.positiveOnly = True

class PearsonIII(Distribuition):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearson3.html
    def __init__(self):
        super().__init__(stats.pearson3, None, distribuition_source=DistribuitionSource.SCIPY)

class LogPearsonIII(TransformDistribuition, PearsonIII):
    def __init__(self):
        PearsonIII.__init__(self)
        TransformDistribuition.__init__(
            self, 
            transform=np.log, 
            inv_transform=np.exp, 
            jacobian_transform=lambda x: 1/x
        )
        