from llama_cpp import Llama
import timeit
import threading

# @timeit
def import_model(model_path):
    llm = Llama(model_path=model_path)
    return llm

# @timeit
def predict(llm):
    output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
    return output

llama_path = ""
llama = import_model(llama_path)

def execute_code():
    result = predict(llama)
    return result

# Create two threads
thread1 = threading.Thread(target=execute_code)
thread2 = threading.Thread(target=execute_code)

# Start the threads
thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

result1 = thread1.get()
result2 = thread2.get()

print(result1)
print(result2)
