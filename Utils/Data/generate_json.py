import json
import os

from world_data import data

directory = os.path.dirname(os.path.abspath(__file__))
filename = "world_data.json"
filepath = os.path.join(directory, filename)

with open(filepath, "w", encoding='utf8') as outfile:
    json.dump(data, outfile, indent=4, ensure_ascii=False)
