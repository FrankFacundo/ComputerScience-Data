import json
import slob

with open('hull_dict.json', 'r', encoding='utf8') as f:
    hull = json.load(f)

with slob.create('hull_dict.slob') as w:
    for key, value in hull.items():
        w.add(value.encode('utf8'), key)

# with slob.create('hull_dict1.slob') as w:
#     for key, value in hull.items():
#         w.add(key, value)
