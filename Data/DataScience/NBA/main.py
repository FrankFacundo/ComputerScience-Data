from fastapi import FastAPI
from pydantic import BaseModel
from model import prediction
from typing import List

app = FastAPI()

# pydantic models
class Player(BaseModel):
    GP: int
    MIN:float
    PTS:float
    FGM:float
    FGA:float
    FG:float
    P3_Made:float
    PA3:float
    P3:float
    FTM:float
    FTA:float
    FT:float
    OREB:float
    DREB:float
    REB:float
    AST:float
    STL:float
    BLK:float
    TOV:float

class Label(BaseModel):
    label: float

# routes
@app.post("/predict",response_model=Label, status_code=200)
def predict_player(player: Player):
    
    return {"label": prediction([player.GP,
                                 player.MIN,
                                 player.PTS,
                                 player.FGM,
                                 player.FGA,
                                 player.FG,
                                 player.P3_Made,
                                 player.PA3,
                                 player.P3,
                                 player.FTM,
                                 player.FTA,
                                 player.FT,
                                 player.OREB,
                                 player.DREB,
                                 player.REB,
                                 player.AST,
                                 player.STL,
                                 player.BLK,
                                 player.TOV])}

@app.post("/predictWithList",response_model=Label, status_code=200)
def predict_playerList(player: List[float]):
    # player = [GP MIN PTS FGM FGA FG%  3P Made 3PA 3P%  FTM FTA FT%  OREB DREB REB AST STL BLK TOV]
    return {"label": prediction(player)}