# Write json from dictionary
import json
d = {'a': 1, 'b': 2, 'c': 3}
with open("file.json", "w") as outfile:
    json.dump(d, outfile)