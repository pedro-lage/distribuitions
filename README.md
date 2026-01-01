# Hidrologia Estatística - Distribuições

Este projeto fornece uma estrutura flexível para ajuste, análise e comparação de distribuições estatísticas, com foco em aplicações hidrológicas. O sistema permite trabalhar tanto com distribuições individuais quanto com seleção local e regional de distribuições.

## Estrutura Principal

### 1. Classe [Distribuition](https://github.com/Bridge-Holding/sistema_hidrologico/blob/hidrologia-estatistica-dev/Distribuitions/distribuition.py)

A classe base para todas as distribuições estatísticas. Ela encapsula métodos para ajuste de parâmetros (fit), cálculo de funções de densidade (PDF), distribuição acumulada (CDF), quantis (PPF), geração de amostras, testes de aderência (Kolmogorov-Smirnov, SSE, AIC, BIC), e visualização gráfica.

- Ajuste de parâmetros: Suporta métodos como momentos (MOM), máxima verossimilhança (MVS), e momentos-L (MML, via lmoments3).
- Visualização: Métodos para plotar PDF, CDF, PPF e histogramas dos dados ajustados.
- Testes de ajuste: Implementa testes como Kolmogorov-Smirnov, SSE, AIC e BIC (estes últimos para distribuições LMOMENTS3).

Exemplo de uso:

```py
from Distribuitions.distribuitions import GenPareto
import numpy as np

xs = np.random.lognormal(mean=5, sigma=0.5, size=200)
distr = GenPareto()
distr.fit(xs, method='mvs')
print(distr.scipy_params)
distr.describe_plot()
distr.fit_test()
```

### 2. Classe [DistribuitionSelector](https://github.com/Bridge-Holding/sistema_hidrologico/blob/hidrologia-estatistica-dev/Distribuitions/distribuitionSelector.py)

Permite comparar e visualizar múltiplas distribuições ajustadas a um mesmo conjunto de dados. Ideal para estudos locais, onde se deseja comparar o ajuste de diferentes distribuições para uma série temporal específica.

- Adição de distribuições: Use appendDistribuition para adicionar instâncias de distribuições.
- Ajuste conjunto: O método fit ajusta todas as distribuições adicionadas aos dados.
- Visualização: O método describePlot plota PDF, CDF e PPF das distribuições ajustadas lado a lado.
- Parâmetros e testes: Métodos para obter parâmetros ajustados (getParams) e resultados dos testes de aderência (fitTest).

Exemplo de uso:

```py
from Distribuitions.distribuitions import Normal, Exponential, Gumbel, GEV
from Distribuitions.distribuitionSelector import DistribuitionSelector

selector = DistribuitionSelector()
selector.appendDistribuition(Normal())
selector.appendDistribuition(Exponential())
selector.appendDistribuition(Gumbel())
selector.appendDistribuition(GEV())

selector.fit(xs, method='mvs')
selector.describePlot()
print(selector.getParams())
print(selector.fitTest())
```

### 2. Classe [RegionalDistribuitionSelector](https://github.com/Bridge-Holding/sistema_hidrologico/blob/hidrologia-estatistica-dev/Distribuitions/regionalDistribuitionSelector.py)

Voltada para análise regional, permite ajustar e comparar distribuições em múltiplos conjuntos de dados (por exemplo, várias estações ou sub-bacias).

- Adição de distribuições: Similar ao DistribuitionSelector, mas para análise regional.
- Ajuste em lote: O método fitData ajusta as distribuições a cada conjunto de dados da região.
- Visualização e estatísticas: Métodos para visualizar boxplots e histogramas dos parâmetros ajustados, além de métricas de ajuste (p-value, SSE, AIC, BIC) para cada distribuição em cada local.
- Análise de parâmetros: O método distribuitionParameters retorna estatísticas agregadas dos parâmetros ajustados regionalmente.

### 2. Classe [DistribuitionSource](https://github.com/Bridge-Holding/sistema_hidrologico/blob/hidrologia-estatistica-dev/Distribuitions/distribuitionSelector.py)

Define a origem/fonte da implementação da distribuição:

- LMOMENTS3: Distribuições implementadas via biblioteca lmoments3, permitindo ajuste por momentos-L.
- SCIPY: Distribuições da biblioteca scipy.stats, geralmente ajustadas por máxima verossimilhança.

## Exemplos de Uso
Ajustando e Visualizando uma Distribuição
```py
from Distribuitions.distribuitions import GenPareto
import numpy as np

xs = np.random.lognormal(mean=5, sigma=0.5, size=200)
distr = GenPareto()
distr.fit(xs, method='mvs')
print(distr.scipy_params)
distr.describe_plot()
```
Comparando Distribuições Locais
```py
from Distribuitions.distribuitions import Normal, Exponential, Gumbel, GEV
from Distribuitions.distribuitionSelector import DistribuitionSelector

selector = DistribuitionSelector()
selector.appendDistribuition(Normal())
selector.appendDistribuition(Exponential())
selector.appendDistribuition(Gumbel())
selector.appendDistribuition(GEV())

selector.fit(xs, method='mvs')
selector.describePlot()
print(selector.getParams())
print(selector.fitTest())
```
Análise Regional
```py
from Distribuitions.distribuitions import Normal, Gumbel
from Distribuitions.regionalDistribuitionSelector import RegionalDistribuitionSelector

regional_selector = RegionalDistribuitionSelector()
regional_selector.appendDistribuition(Normal())
regional_selector.appendDistribuition(Gumbel())

# xss: lista de arrays, cada um com dados de uma região
regional_selector.fitData(xss)
regional_selector.goodnessOfFitResults()
regional_selector.distribuitionParameters()
```