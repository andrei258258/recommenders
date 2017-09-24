
# coding: utf-8

# # Sistema de Recomendação
# ## Baseado em Matrix Factorization SVD

# Autores: Andrei Donati, Angelo Baruffi e Luís Felipe Pelison

# O DataSet utilizado para este código é o MovieLens (https://grouplens.org/datasets/movielens/). 
# A fim de teste, foi adicionado entradas referentes ao desenvolvedor. 

# # Bibliotecas python (3)

# In[221]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
get_ipython().magic('matplotlib inline')
#import os
#os.chdir('c:\\Andrei\\recommender-systems\\dataset')


# # Importação dos dados

# ## Ratings

# In[222]:

ratings_df = pd.read_csv("./dataset/ratings.csv", sep=",")
ratings_df.head()


# Temos o id do usuário (*userId*), do filme (*movieId*), a nota dada pelo usuário ao filme (*rating*) e o horário que essa nota foi dada (*timestamp*)

# Porém, não precisaremos do horário aqui, então vamos excluir a coluna *timestamp*

# In[223]:

ratings_df = ratings_df.iloc[:,0:3]
ratings_df.head()


# ### Separação em teste e treino

# In[224]:

## Vamos separar os dados em 2 conjuntos: teste e treino
ratings_df, ratings_df_test = train_test_split(
                                ratings_df, test_size=0.05, random_state=42)

## Visualizando a quantidade de dados de cada conjunto
ratings_df_test.shape[0], ratings_df.shape[0]


# ## Movies

# In[225]:

movies_df = pd.read_csv("./dataset/movies.csv", sep=",")
movies_df.head()


# In[226]:

movies_df.info()


# # Adicionando dados pessoais

# In[227]:

#Quantos usuários existem?

num_users  = np.max(ratings_df['userId']) +1
print("Number of users {}".format(num_users))


# Adicionando os meus dados

# In[228]:

##Agora, vou criar um novo usuário, de id 672 (O próximo da lista)
##A classificação do novo usuário será uma tabela onde a primeira coluna é o id 672, a segunda é o id do filme 
## e a terceira a nota. > [userId, movieId, rating]

my_ratings= pd.DataFrame([[672, 1, 3.5 ], [672, 135861, 4.5], [672, 133824, 1], [672, 130634, 5], 
                          [672, 116797, 4.5], [672, 114662, 4], [672, 112897, 4], [672, 112497, 2],
                          [672, 112370, 3.5], [672, 112183, 5], [672, 110553, 3.5], [672, 109673, 4.5],
                          [672, 108932, 1], [672, 50189, 4], [672, 59784, 3.5], [672, 61123, 1], 
                          [672, 61160, 2], [672, 62999, 4], [672, 63515, 4.5], [672, 67923, 4.5],
                          [672, 68157, 5]], columns=['userId','movieId','rating']
                        ) 

print('Visualizando os filmes que gosto')
## Join das minhas avaliações com o título do filme e os gêneros, somente para visualização
my_ratings.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').sort_values(by=['rating'], ascending=False).head(10)


# In[229]:

my_ratings.info()


# Une os datasets

# In[230]:

ratings_df = ratings_df.append(my_ratings)
ratings_df[ratings_df['userId'] == 672]


# ## Matriz de Users x Filmes, com os valores das notas dadas

# In[231]:

R_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
R_df[-5:]


# ### Visualização de algumas informações importantes

# Entradas não nulas da matriz

# In[232]:

print('% entradas não nulas: {:0.4f} %'.format(100*(np.count_nonzero(R_df>0.0)/(R_df.shape[0]*R_df.shape[1]))) )


# In[233]:

R_df[R_df>0.0].count(axis = 0).sort_values(ascending= False)[:30].plot(figsize=(15,5), kind='bar', title='Quantidade de ratings por filme (top 30)');


# In[234]:

R_df[R_df>0.0].count(axis = 1).sort_values(ascending= False)[:30].plot(figsize=(15,5), kind='bar', title='Quantidade de ratings por usuário (top 30)');


# In[235]:

R_df[R_df>0.0].mean(axis = 1).sort_values(ascending= False).plot(figsize=(17,4), kind='bar', title='Nota Média de cada usuário (geral)');


# # Modelo: Algoritmo SVD

# Substitui os valores nulos por 0 e transforma a matriz de reviews em uma matriz com uma diferença da média do usuário
# 

# In[236]:

R = R_df.as_matrix()
R.shape, R


# In[237]:

user_ratings_mean = np.mean(R, axis=1)
user_ratings_mean, user_ratings_mean.shape


# In[238]:

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
R_demeaned, R_demeaned.shape


# ### Aplica a transformação SVD

# In[239]:

U, sigma, Vt = svds(R_demeaned, k = 80)

sigma = np.diag(sigma)


# #### Reconstroi a matrix de ratings

# In[240]:

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings = all_user_predicted_ratings + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
preds_df[-5:]


# In[241]:

del U, sigma, Vt,  R, user_ratings_mean, all_user_predicted_ratings


# Define alguns parâmetros

# In[242]:

userId= 672
num_recommendations= 10


# #### Filtra os filmes já assistidos

# In[243]:

user_row_number = userId - 1 

#pega as filmes que o usuário já assistiu
sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) 
    
user_data = ratings_df[ratings_df['userId'] == (userId)]
already_rated = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                sort_values(['rating'], ascending=False)
            )

print('Usuário {0} já deixou seu rating para {1} filmes.'.format(userId, already_rated.shape[0]) )
  


# # Recomendações

# In[244]:

predictions = (movies_df[~movies_df['movieId'].isin(already_rated['movieId'])].
                     merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                   left_on = 'movieId', right_on = 'movieId').
                   rename(columns = {user_row_number: 'Predictions'}).
                   sort_values('Predictions', ascending = False).
                   iloc[:num_recommendations, :-1]
                  )

print('Recomendando os top {0} filmes ainda não vistos pelo usuário.'.format(num_recommendations))


# In[245]:

predictions


# ## Comparando
# ### com os filmes que melhor foram avaliados por mim:

# In[246]:

already_rated

