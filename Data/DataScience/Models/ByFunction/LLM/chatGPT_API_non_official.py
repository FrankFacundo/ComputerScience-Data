from revChatGPT.V1 import Chatbot
import time

chatbot = Chatbot(config={
  "email": "frank.facundo@telecom-paris.fr",
  "password": "telecom2021paris"
})

prompt = "again, how many beaches does portugal have?"
response = ""

init = time.time()
for data in chatbot.ask(prompt):
    print(data["message"])
    print("***")
    response = data["message"]
print("Total time: ", time.time() - init)
print(response)