# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pickle import dump

# %% [markdown]
# # Dataset

# %%
data = pd.read_csv("./app/data/ATP.csv")

# %% [markdown]
# # Preprocessing


# %%
data_filtered = data.drop(columns=["best_of", "draw_size","loser_name","winner_name","loser_entry","winner_entry","loser_seed","winner_seed", "match_num", "minutes", "round", "score", "surface", "tourney_date","tourney_name", "tourney_id", "tourney_level"]).dropna()


# %%
dict_hand = {k:i for i,k in enumerate(np.unique(pd.concat([data_filtered["loser_hand"],data_filtered["winner_hand"]],axis=0)))}
dict_ioc = {k:i for i,k in enumerate(np.unique(pd.concat([data_filtered["loser_ioc"],data_filtered["winner_ioc"]],axis=0)))}

dump([dict_hand, dict_ioc], open('app/dicts.pkl', 'wb'))


# %%
data_filtered["winner_hand"] = data_filtered["winner_hand"].apply(lambda x: dict_hand[x])
data_filtered["loser_hand"] = data_filtered["loser_hand"].apply(lambda x: dict_hand[x])
data_filtered["winner_ioc"] = data_filtered["winner_ioc"].apply(lambda x: dict_ioc[x])
data_filtered["loser_ioc"] = data_filtered["loser_ioc"].apply(lambda x: dict_ioc[x])



# %%
data_filtered.columns = ['first_1stIn', 'first_1stWon', 'first_2ndWon', 'first_SvGms', 'first_ace',
       'first_bpFaced', 'first_bpSaved', 'first_df', 'first_svpt', 'first_age', 'first_hand',
       'first_ht', 'first_id', 'first_ioc', 'first_rank', 'first_rank_points',
       'second_1stIn', 'second_1stWon',
       'second_2ndWon', 'second_SvGms', 'second_ace', 'second_bpFaced', 'second_bpSaved', 'second_df',
       'second_svpt', 'second_age', 'second_hand', 'second_ht', 'second_id',
       'second_ioc', 'second_rank', 'second_rank_points']
data_filtered["class"]=1

# %% [markdown]
# # Increase data

# %%
data_filtered_switched = data_filtered.copy(deep=True)
cols = data_filtered_switched.columns.tolist()
cols_switched = cols[16:32] + cols[0:16]
data_filtered_switched = data_filtered_switched[cols_switched]


# %%
data_filtered_switched.columns = ['first_1stIn', 'first_1stWon', 'first_2ndWon', 'first_SvGms', 'first_ace',
       'first_bpFaced', 'first_bpSaved', 'first_df', 'first_svpt', 'first_age', 'first_hand',
       'first_ht', 'first_id', 'first_ioc', 'first_rank', 'first_rank_points',
       'second_1stIn', 'second_1stWon',
       'second_2ndWon', 'second_SvGms', 'second_ace', 'second_bpFaced', 'second_bpSaved', 'second_df',
       'second_svpt', 'second_age', 'second_hand', 'second_ht', 'second_id',
       'second_ioc', 'second_rank', 'second_rank_points']
data_filtered_switched["class"]=0


# %% [markdown]
# # Prepare final dataset

# %%
dataset = pd.concat([data_filtered,data_filtered_switched],axis=0)


# %%
dataset = dataset.sample(frac=1)
X, y = dataset.iloc[:,:-1], dataset["class"]


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dump(scaler, open('app/scaler.pkl', 'wb'))

# %% [markdown]
# # Train LGBMClassifier

# %%
clf=lgb.LGBMClassifier(boosting_type='gbdt',num_leaves=200,learning_rate=0.01,n_estimators=1000,reg_alpha=1.0,reg_lambda=1.0)
print('Training...')
clf.fit(X_train,y_train)
print('Training finished')
clf.booster_.save_model('app/lgbr_model.txt')

# %% [markdown]
# # Accuracy

# %%
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_pred,y_test)
print(f"Accuracy: {accuracy}")

# %% [markdown]
# # Feature Importance

# %%
plt.bar(np.arange(len(X.columns)),clf.feature_importances_)


