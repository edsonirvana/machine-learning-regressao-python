import pandas as pd
import statsmodels.formula.api as smf

# Carregando o dataset
df = pd.read_csv('dados/pesos.csv')

# Criando o Modelo de Regressão
estimativa = smf.ols(formula = 'Peso ~ Idade', data = df)

# Treinando o Modelo de Regressão
modelo = estimativa.fit()

# Imprimindo o resumo do modelo
print(modelo.summary())
