import neptune.new as neptune

# Changes all values:
#  - <USER_ID> with your Neptune.ai user ID,
#  - <PROJECT> with your Neptune.ai Experiment Tracking Project,
#  - <API_TOKEN> with your Neptune.ai API Token (Click on top right icon in header bar, and then "Get your API Token").
run = neptune.init(
    project="<USER_ID>/<PROJECT>",
    api_token="<API_TOKEN>",
)

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].log(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()
