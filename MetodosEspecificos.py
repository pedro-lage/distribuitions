import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from DownloadDados import *
from Distribuitions.distribuitions import *
from Distribuitions.distribuitionSelector import *
from Distribuitions.regionalDistribuitionSelector import *
import os

'''//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''

#%%
class Q710:
    
    """IMPORTANTE:
    
    1 - Essas funções descritas abaixo levam como pramissas os seguintes:
            - Dados de entrada são dados de séries históricas diários, caso não sejam as probabilidades e valores relacionados
              a cada tempo de retorno estipulado estarão consequentemente incorretos.
            - Os dados devem ser ou um Dataframe, um .csv ou um .xlsx e devem ser necessáriamentes no formato: Data(Primeira coluna) e Valores(Segunda coluna)
    2 - As distribuições que são ajustadas são as seguintes: Gumbel(para mínimos), Weibull(para mínimos) --(Outras distribuições
        devem ser acrecentadas no futuro)--.
        
        É muito importante que o engenheiro responsável em projetos de dimensionamento verifique se a distribuição ajustada pode ser realmente útilizada com 
        base em outros parâmetros, este algorítimo apenas ajusta as distribuições e realiza alguns tipos de teste de aderência, mas isso sozinho não diz se
        uma distribuição é realmente válida para determinados casos.
        
    """
    @staticmethod
    def calcular_m7_min_anual(df, maximo_falhas):
        """
        Calcula a média móvel de 7 dias e o menor valor anual da média móvel para um DataFrame.

        Parâmetros:
        df (pd.DataFrame): DataFrame com pelo menos duas colunas, a primeira sendo datas e a segunda valores.

        Retorna:
        pd.DataFrame: DataFrame com o menor valor anual da média móvel de 7 dias.
        """
        df = df.copy()  # Corrigido para chamar a função copy corretamente
        # Garante que os dados sejam diários (se forem horários, calcula a média diária)
        if not isinstance(df.index, pd.DatetimeIndex):
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')
            df.set_index(df.columns[0], inplace=True)

        df = df.resample('D').mean()


        # Calcula a média móvel de 7 dias
        df['M7'] = df.iloc[:, 0].rolling(window=7, min_periods=7).mean()
        print(df)
        # Criar um novo DataFrame com anos válidos
        df['Ano'] = df.index.year
        anos_validos = []

        df_anual = df.groupby('Ano')
        for ano, grupo in df_anual:
            total_dias = len(grupo)
            dias_falha = grupo.iloc[:, 0].isna().sum()
            percentual_falha = (dias_falha / total_dias) * 100

            if maximo_falhas is not None and percentual_falha <= maximo_falhas:
                anos_validos.append(ano)

        # Filtra apenas os anos válidos
        df_filtrado = df[df['Ano'].isin(anos_validos)]

        # Calcula o menor M7 para cada ano
        df_m7_min_anual = df_filtrado.groupby('Ano')['M7'].min().reset_index()
        df_m7_min_anual.rename(columns={'M7': 'M7_min_anual'}, inplace=True)
        df_m7_min_anual = df_m7_min_anual.sort_values(by='M7_min_anual', ascending=True)
        print(df_m7_min_anual)
        return df_m7_min_anual

    @staticmethod
    def CalculoQ710(df, maximo_falhas):
        # Calcula o menor M7 para cada ano
        df_m7_min_anual = Q710.calcular_m7_min_anual(df, maximo_falhas)

        # Criando instâncias das distribuições GumbelMin e Weibull
        gumbel_min_dist = GumbelMin()
        weibull_min_dist = Weibull()

        # Criando instância do DistribuitionSelector
        selector = DistribuitionSelector()

        # Adicionando as distribuições ao selector
        selector.appendDistribuition(gumbel_min_dist)
        selector.appendDistribuition(weibull_min_dist)

        # Ajustando as distribuições aos dados
        selector.fit(df_m7_min_anual['M7_min_anual'], method='mml')

        # Exibindo os parâmetros ajustados
        params = selector.getParams()
        gumbel_params = params[0]
        weibull_params = params[1]
        print(gumbel_params, weibull_params)

        # Plotando os gráficos das distribuições ajustadas
        selector.describePlot()

        # Exibindo os parâmetros ajustados
        params = selector.getParams()
        gumbel_params = params[0]
        weibull_params = params[1]

        # Exibindo os resultados dos testes de ajuste
        fit_test_results = selector.fitTest()
        gumbel_fit_test = fit_test_results[0]
        weibull_fit_test = fit_test_results[1]

        # Calculando o valor de .ppf(0.1) para cada distribuição
        Q710_Weibull = weibull_min_dist.ppf(0.1)
        Q710_Gumbel = gumbel_min_dist.ppf(0.1)

        return gumbel_params, weibull_params, gumbel_fit_test, weibull_fit_test, Q710_Weibull, Q710_Gumbel
    

#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class CurvaPermanencia:
    def __init__(self, df):
        self.df = df
        self.df_consistido = None
        self.vazao_media = None
        self.vazao_minima = None
        self.vazao_maxima = None
        self.q95 = None
        self.q50 = None

    def calcular_vazoes_de_referencia(self):
        """
        Calcula as vazões de referência (média, mínima, máxima e Q95) considerando apenas dados com nível de consistência 2.
        Também plota a curva de permanência da vazão.
        
        :return: Dicionário com os valores de vazão média, mínima, máxima, Q95 e Q50.
        """
        df = self.df

        if df.empty or 'Vazao' not in df.columns or 'NivelConsistencia' not in df.columns:
            print("⚠️ DataFrame inválido ou sem dados suficientes para calcular as vazões de referência.")
            return None

        # Removendo espaços em branco e substituindo valores vazios por NaN antes da conversão
        df['Vazao'] = df['Vazao'].astype(str).str.strip().replace("", np.nan)
        df['NivelConsistencia'] = df['NivelConsistencia'].astype(str).str.strip().replace("", np.nan)

        # Converter para numérico
        df['Vazao'] = pd.to_numeric(df['Vazao'], errors='coerce')
        df['NivelConsistencia'] = pd.to_numeric(df['NivelConsistencia'], errors='coerce')

        # Verificar a quantidade de NaNs após conversão (debug)
        vazao_nans = df['Vazao'].isna().sum()
        nivel_nans = df['NivelConsistencia'].isna().sum()
        print(f"🔍 Valores NaN após conversão: Vazão={vazao_nans}, Nível Consistência={nivel_nans}")

        # Filtrar apenas dados com nível de consistência igual a 2
        self.df_consistido = df[df['NivelConsistencia'] == 2]

        if self.df_consistido.empty:
            print("⚠️ Nenhum dado disponível com nível de consistência 2 para calcular as vazões de referência.")
            return None

        # Calculando os valores de referência
        self.vazao_media = self.df_consistido['Vazao'].mean()
        self.vazao_minima = self.df_consistido['Vazao'].min()
        self.vazao_maxima = self.df_consistido['Vazao'].max()
        self.q95 = self.df_consistido['Vazao'].quantile(0.05)  # Percentil 5%
        self.q50 = self.df_consistido['Vazao'].quantile(0.50)  # Mediana

        return {
            "Vazão Média": self.vazao_media,
            "Vazão Mínima": self.vazao_minima,
            "Vazão Máxima": self.vazao_maxima,
            "Q95": self.q95,
            "Q50": self.q50
        }

    def plotar_curva_permanencia(self):
        """ Plota a curva de permanência da vazão com o eixo y em escala logarítmica. """
        if self.df_consistido is None:
            print("⚠️ Nenhum dado disponível para plotar a curva de permanência.")
            return None

        # Ordenar os valores para a curva de permanência
        vazoes_ordenadas = self.df_consistido['Vazao'].dropna().sort_values(ascending=False)
        percentis = np.arange(1, len(vazoes_ordenadas) + 1) / len(vazoes_ordenadas)

        # Criando a curva de permanência
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=percentis * 100,
            y=vazoes_ordenadas,
            mode='lines',
            name='Curva de Permanência'
        ))

        # Linha de referência para Q95
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[self.q95, self.q95],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'Q95: {self.q95:.2f} m³/s'
        ))

        # Linha de referência para Q50 (mediana)
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[self.q50, self.q50],
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name=f'Q50: {self.q50:.2f} m³/s'
        ))

        # Definindo os ticks do eixo y manualmente, excluindo os valores de 0 a 9
        y_ticks = [10**i for i in range(int(np.floor(np.log10(vazoes_ordenadas.min()))), int(np.ceil(np.log10(vazoes_ordenadas.max()))) + 1)]

        fig.update_layout(
            title='Curva de Permanência de Vazão',
            xaxis_title='Percentil (%)',
            yaxis_title='Vazão (m³/s)',
            yaxis_type='log',  # Escala logarítmica no eixo y
            yaxis=dict(
                tickvals=y_ticks,
                ticktext=[str(tick) for tick in y_ticks]
            ),
            template='plotly_white',
            showlegend=True
        )

        fig.show()

# Exemplo de uso
# curva_permanencia = CurvaPermanencia(df_teste)
# vazoes_referencia = curva_permanencia.calcular_vazoes_de_referencia()
# curva_permanencia.plotar_curva_permanencia()
# curva_permanencia.plotar_cotagrama()
# curva_permanencia.plotar_fluviograma()
# print(vazoes_referencia)


class Vazoes:
    def __init__(self, df):
        self.df = df
        self.df_consistido = None
        self.vazao_media = None
        self.vazao_minima = None
        self.vazao_maxima = None
        self.vazoes_medias_mensais = None
        self.vazoes_medias_anuais = None
        self.vazao_media_longo_termo = None
 
    def calcular_vazao_minima(self, df):
        df['Vazao'] = pd.to_numeric(df['Vazao'], errors='coerce')
        vazao_minima = df['Vazao'].min()
        return vazao_minima
 
    def calcular_vazao_maxima(self, df):
        df['Vazao'] = pd.to_numeric(df['Vazao'], errors='coerce')
        vazao_maxima = df['Vazao'].max()
        return vazao_maxima
 
    def calcular_vazoes_medias_mensais(self, df):
        """
        Calcula as vazões médias mensais para todos os anos disponíveis
       
        Args:
            df (pd.DataFrame): DataFrame contendo as colunas 'Data' e 'Vazao'
           
        Returns:
            pd.DataFrame: DataFrame com as vazões médias mensais (Mês e VazaoMedia)
        """
        # Converter colunas para os tipos adequados
        df['Data'] = pd.to_datetime(df['Data'])
        df['Vazao'] = pd.to_numeric(df['Vazao'], errors='coerce')
       
        # Extrair mês e ano
        df['Mes'] = df['Data'].dt.month
        df['Ano'] = df['Data'].dt.year
       
        # Calcular média mensal agrupando por mês
        vazoes_mensais = df.groupby('Mes')['Vazao'].mean().reset_index()
        vazoes_mensais.columns = ['Mes', 'VazaoMedia']
       
        return vazoes_mensais
 
    def calcular_vazoes_medias_anuais(self, df):
        """
        Calcula as vazões médias anuais para todos os anos disponíveis
       
        Args:
            df (pd.DataFrame): DataFrame contendo as colunas 'Data' e 'Vazao'
           
        Returns:
            pd.DataFrame: DataFrame com as vazões médias anuais (Ano e VazaoMedia)
        """
        # Converter colunas para os tipos adequados
        df['Data'] = pd.to_datetime(df['Data'])
        df['Vazao'] = pd.to_numeric(df['Vazao'], errors='coerce')
       
        # Extrair ano
        df['Ano'] = df['Data'].dt.year
       
        # Calcular média anual agrupando por ano
        vazoes_anuais = df.groupby('Ano')['Vazao'].mean().reset_index()
        vazoes_anuais.columns = ['Ano', 'VazaoMedia']
       
        return vazoes_anuais
 
    def calcular_vazao_media_longo_termo(self, df):
        """
        Calcula a vazão média de longo termo (média de todas as vazões disponíveis)
       
        Args:
            df (pd.DataFrame): DataFrame contendo a coluna 'Vazao'
           
        Returns:
            float: Valor da vazão média de longo termo
        """
        df['Vazao'] = pd.to_numeric(df['Vazao'], errors='coerce')
        vazao_media = df['Vazao'].mean()
        return vazao_media
    
    
    


class Analise_de_Maximos:
    
    """IMPORTANTE:
    
    1 - Essas funções descritas abaixo levam como pramissas os seguintes:
            - Dados de entrada são dados de séries históricas diários, caso não sejam as probabilidades e valores relacionados
              a cada tempo de retorno estipulado estarão consequentemente incorretos.
            - Os dados devem ser ou um Dataframe, um .csv ou um .xlsx e devem ser necessáriamentes no formato: Data(Primeira coluna) e Valores(Segunda coluna)
    2 - As distribuições que são ajustadas são as seguintes: Gumbel(para máximos), GEV, Weibull(para máximos) e a distribuição Gamma --(Outras distribuições
        devem ser acrecentadas no futuro)--.
        
        É muito importante que o engenheiro responsável em projetos de dimensionamento verifique se a distribuição ajustada pode ser realmente útilizada com 
        base em outros parâmetros, este algorítimo apenas ajusta as distribuições e realiza alguns tipos de teste de aderência, mas isso sozinho não diz se
        uma distribuição é realmente válida para determinados casos.
        
        Vale notar que a distribuição de Weibull para máximos foi concebida com base na Weibull de mínimos dois parâmetros, ou seja, o parâmetro de posição é dado
        como igual a zero, limitando assim seu limite inferior, passo importante para a Weibull de mínimos, da qual essa foi derivada.
        
      
    """
    
    
    def maximos_anuais_hidrologicos(self, input_data, mes_inicio_ano_hidro=10, mes_fim_ano_hidro=9, 
                                  maximo_falhas=0.1, meses_criticos=None, **kwargs):
        """
        Calcula os máximos anuais hidrológicos filtrando anos incompletos.
        
        Parâmetros:
        -----------
        input_data : DataFrame ou str
            Fonte de dados (DataFrame, caminho para Excel/CSV)
        mes_inicio_ano_hidro : int (opcional)
            Mês de início do ano hidrológico (padrão=10)
        mes_fim_ano_hidro : int (opcional)
            Mês de fim do ano hidrológico (padrão=9)
        maximo_falhas : float (opcional)
            % mínimo de dados para considerar o ano (0-1, padrão=0.1)
        meses_criticos : list (opcional)
            Lista de meses onde falhas são críticas (ex: [8,9,10,11,12,1,2])
            Se None, considera todos os meses
        **kwargs : 
            Parâmetros adicionais para leitura de arquivos
            
        Retorna:
        --------
        DataFrame com máximos anuais e anos hidrológicos
        """
        # 1. Carregar dados (mesmo código anterior)
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, str):
            ext = os.path.splitext(input_data)[1].lower()
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(input_data, **kwargs)
            elif ext == '.csv':
                df = pd.read_csv(input_data, **kwargs)
            else:
                raise ValueError("Formato inválido. Use Excel ou CSV")
        else:
            raise TypeError("Entrada deve ser DataFrame ou caminho de arquivo")

        # Verificar estrutura
        if len(df.columns) < 2:
            raise ValueError("Dados devem ter colunas Data e Valor")

        # 2. Processar dados
        data_col, valor_col = df.columns[0], df.columns[1]
        df[data_col] = pd.to_datetime(df[data_col])
        df = df.dropna(subset=[valor_col])

        # 3. Calcular ano hidrológico
        df['AnoHidrologico'] = df[data_col].apply(
            lambda x: f"{x.year}-{x.year+1}" if x.month >= mes_inicio_ano_hidro else f"{x.year-1}-{x.year}"
        )
        df['Mes'] = df[data_col].dt.month

        # 4. Filtrar anos com poucos dados nos meses críticos
        if meses_criticos is not None:
            # Converter meses para inteiros (caso usuário passe strings)
            meses_criticos = [int(m) for m in meses_criticos]
            
            # Criar máscara para meses críticos
            mask_meses_criticos = df['Mes'].isin(meses_criticos)
            
            # Contar dados válidos apenas nos meses críticos por ano
            contagem_critica = df[mask_meses_criticos].groupby('AnoHidrologico')[valor_col].count()
            
            # Total esperado de dias nos meses críticos (aproximação)
            dias_esperados_criticos = len(meses_criticos) * 30  # ~30 dias por mês
            
            # Filtrar anos com dados insuficientes nos meses críticos
            anos_validos = contagem_critica[
                contagem_critica >= (dias_esperados_criticos * maximo_falhas)
            ].index
        else:
            # Comportamento original - considerar todos os meses
            contagem_por_ano = df.groupby('AnoHidrologico')[valor_col].count()
            dias_esperados = 365  # Ano completo
            anos_validos = contagem_por_ano[contagem_por_ano >= (dias_esperados * maximo_falhas)].index

        df = df[df['AnoHidrologico'].isin(anos_validos)]

        # 5. Calcular máximos anuais
        maximos = df.groupby('AnoHidrologico')[valor_col].max().reset_index()
        maximos.columns = ['Ano Hidrológico', 'Máximo Anual']

        # 6. Ordenar resultados
        maximos['_AnoInicio'] = maximos['Ano Hidrológico'].str.split('-').str[0].astype(int)
        maximos = maximos.sort_values('_AnoInicio').drop('_AnoInicio', axis=1)

        return maximos
    
    
    def CalculoMaximas(self, input_data, mes_inicio_ano_hidro=10, mes_fim_ano_hidro=9, maximo_falhas=0.1,  meses_criticos=None):
        # Calcula o menor M7 para cada ano
        df_maximas_anual = self.maximos_anuais_hidrologicos(input_data, mes_inicio_ano_hidro, mes_fim_ano_hidro, maximo_falhas,  meses_criticos) 

        # Criando instâncias das distribuições GumbelMin e Weibull
        gumbel_max_dist = Gumbel()
        weibull_max_dist = Weibull_Max()
        GEV_dist = GEV()
        Gama_dist = Gama()
        #Frechet_dist = Frechet()
        #LogNormal_dist = LogNormal()

        # Criando instância do DistribuitionSelector
        selector = DistribuitionSelector()

        # Adicionando as distribuições ao selector
        selector.appendDistribuition(gumbel_max_dist)
        selector.appendDistribuition(weibull_max_dist)
        selector.appendDistribuition(GEV_dist)
        selector.appendDistribuition(Gama_dist)
        #selector.appendDistribuition(Frechet_dist)
        #selector.appendDistribuition(LogNormal_dist)

        # Ajustando as distribuições aos dados
        selector.fit(df_maximas_anual['Valor'], method='mvs')

        # Plotando os gráficos das distribuições ajustadas
        selector.describePlot()

        # Exibindo os parâmetros ajustados
        params = selector.getParams()
        gumbel_max_dist_params = params[0]
        weibull_max_distparams = params[1]
        GEV_dist_params = params[2]
        Gama_dist_params = params[3]
        #Frechet_dist_params = params[4]
        #LogNormal_dist_params = params[5]

        # Exibindo os resultados dos testes de ajuste
        fit_test_results = selector.fitTest()
        gumbel_max_dist_fit_test = fit_test_results[0]
        weibull_max_dist_fit_test = fit_test_results[1]
        GEV_dist_fit_test = fit_test_results[2]
        Gama_dist_fit_test = fit_test_results[3]
        #Frechet_dist_fit_test = fit_test_results[4]
        #LogNormal_dist_fit_test = fit_test_results[5]

        # Calculando o valor para cada tempo de retorno em anos
        tempo_retorno = [2, 5, 10, 25, 50, 100, 200, 500, 1000, 10000]
        # Criar listas vazias para armazenar os resultados
        PPF_gumbel_max__list = []
        PPF_weibull_max_list = []
        PPF_GEV_list = []
        PPF_Gama_list = []

        for i in tempo_retorno:
            probabilidade = 1/i
            
            # Calcular os valores
            PPF_gumbel_max = gumbel_max_dist.PPF(probabilidade, *gumbel_max_dist_params)
            PPF_weibull_max = weibull_max_dist.PPF(probabilidade, *weibull_max_distparams)
            PPF_GEV = GEV_dist.PPF(probabilidade, *GEV_dist_params)
            PPF_Gama = Gama_dist.PPF(probabilidade, *Gama_dist_params)
            
            # Adicionar às listas
            PPF_gumbel_max__list.append(PPF_gumbel_max)
            PPF_weibull_max_list.append(PPF_weibull_max)
            PPF_GEV_list.append(PPF_GEV)
            PPF_Gama_list.append(PPF_Gama)



        return (gumbel_max_dist_params,
                weibull_max_distparams,
                GEV_dist_params,
                Gama_dist_params,
                gumbel_max_dist_fit_test,
                weibull_max_dist_fit_test,
                GEV_dist_fit_test,
                Gama_dist_fit_test,
                PPF_gumbel_max__list,
                PPF_weibull_max_list,
                PPF_GEV_list,
                PPF_Gama_list
        )

# %%
