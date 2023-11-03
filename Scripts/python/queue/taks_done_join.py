"""
Here's how the coordination between task_done() and join() works:
When you enqueue an item using put(), an internal counter within the queue object is incremented. 
This counter tracks the number of unfinished tasks.
When you call task_done() after processing an item, that counter is decremented.
The join() method simply waits until the counter reaches zero. 
This means that join() is waiting for the number of task_done() calls to match the number of put() calls.
"""

from queue import Queue
from threading import Thread, current_thread
import time

def worker(q):
    while not q.empty():
        task = q.get()
        print(f"Thread {current_thread().name} is processing task: {task}")
        time.sleep(1)  # Simulate task processing by sleeping for 1 second
        q.task_done()  # Signal that the task is done
        print(f"Thread {current_thread().name} has finished task: {task}")

# Create a queue and add some tasks
task_queue = Queue()
for task_index in range(5):
    task_queue.put(f"Task {task_index}")

# Start some worker threads
for i in range(2):  # Let's start 2 worker threads
    t = Thread(target=worker, args=(task_queue,))
    t.daemon = True  # Allows the program to exit even if threads are blocked
    t.start()

# Wait for all tasks to be completed
task_queue.join()
print("All tasks have been processed.")
