import numpy as np
import warnings as wr
from sklearn import neural_network as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
import pandas as pd


resume_df = pd.read_csv("/content/dataSet.csv")
# taxa de aprendizado
lr = 0.1

#criar classificação de acordo com o tier_score
df_tier = pd.DataFrame([
                       [0,'F',  0.000000,  0.138720], 
                       [1,'E', 0.138720,  0.27744],
                       [2,'D', 0.27744,  0.41616],
                       [3,'C', 0.41616,  0.55488],
                       [4,'B', 0.55488,  0.69360],
                       [5,'A', 0.69360,  0.83232],
                       [6,'S', 0.83232,  0.97104],
                       [7,'SS', 0.97104, 1.10977]
                       ], 
                       index=range(0, 8), 
                       columns=['Classe', 'Tier', 'De', 'Ate'])

resume_df['tier'] = -1
for i_df, r_df in resume_df.iterrows():
    for i_df_imc, r_df_imc in df_tier.iterrows():
        if (r_df['tier_score'] >= r_df_imc['De']) and (r_df['tier_score'] < r_df_imc['Ate']):
            resume_df.at[i_df, 'tier'] = df_tier.at[i_df_imc, 'Classe']

#df para a decision tree
columns_X = resume_df[["avg_score","wins_percent_weighted"]]
columns_Y = resume_df[["tier"]]
df_X = columns_X.copy()
df_y = columns_Y.copy()

#separando 65% do dataset para treino e 35% para teste
X_train, X_teste, Y_train, Y_teste = train_test_split(df_X, df_y, test_size=0.35, random_state=42)
         
mlp = nn.MLPClassifier(hidden_layer_sizes=(90,), max_iter=1048, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=lr)

# treino
print('#################### EXECUCAO ####################')
print('Treinamento') 
with wr.catch_warnings():
    wr.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, Y_train)

# teste
print('Testes') 
Y = mlp.predict(X_teste)

# resultado 
print('Resultado procurado') 
print(Y_train)
print("Score de treino: %f" % mlp.score(X_train, Y_train))
print('Resultado encontrado') 
print(Y)
print("Score do teste: %f" % mlp.score(X_teste, Y_teste))
