import lightgbm as lgb
import pandas as pd
from pickle import load

model = lgb.Booster(model_file='app/lgbr_model.txt')
scaler = load(open('app/scaler.pkl', 'rb'))
dicts = load(open('app/dicts.pkl', 'rb'))
dict_hand = dicts[0]
dict_ioc = dicts[1]
index_hand = 10
index_ioc = 13

def getPayloadAsVector(payload):
    player1 = payload.player1.split(', ')
    player1[index_hand] = dict_hand[player1[index_hand]]
    player1[index_ioc] = dict_ioc[player1[index_ioc]]
    print(player1)

    player2 = payload.player2.split(', ')
    player2[index_hand] = dict_hand[player2[index_hand]]
    player2[index_ioc] = dict_ioc[player2[index_ioc]]
    print(player2)

    new_payload = player1 + player2
    new_payload_df = pd.DataFrame(data=[new_payload])
    new_payload = scaler.transform(new_payload_df)
    return new_payload


def predict(payload):
    new_payload = getPayloadAsVector(payload)
    y_pred=round(model.predict(new_payload)[0])
    return y_pred