# %%
import json
import re

import whisper
from glob import glob

# %%
model = whisper.load_model("large")
# print(whisper.available_models())
# ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']

# %%
base_path = "7-CaseInterviewVideos"

for file in glob(base_path + "/*.mp3"):
    print("file", file)
    path_output = file[:-4] + ".json"
    print("path_output", path_output)
    result = model.transcribe(file)
    with open(path_output, "w") as outfile:
        json.dump(result, outfile)


# %%



