import pandas as pd

from world_data import data

gdp_economies = data["Business and economics"]["The world economy"]["GDP Economies"]["data"]

df_gdp = pd.DataFrame.from_dict(gdp_economies)
df_gdp.head()