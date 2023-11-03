"""
Practice Queue Python object
"""

import threading
from queue import Queue, Full


def attempt_put(queue, item, timeout):
    try:
        print(
            f"Attempting to put '{item}' into the queue with a timeout of {timeout} seconds."
        )
        queue.put(item, block=True, timeout=timeout)
        print(f"Successfully put '{item}' into the queue.")
    except Full:
        print(
            f"Timeout occurred. Could not put '{item}' into the queue within {timeout} seconds."
        )

def list_to_queue(list_strings):
    timeout = 2
    queue = Queue(maxsize=0)
    threads = []

    for word in list_strings:
        thread = threading.Thread(target=attempt_put, args=(queue, word, timeout))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return queue

def queue_to_list(queue_strings):
    timeout = 2
    items = []
    while not queue_strings.empty():
        items.append(queue_strings.get(timeout=timeout))
    return items

def print_queue(q):
    # Acquire the mutex for thread-safe access to the queue's internal list
    with q.mutex:
        queue_list = list(q.queue)

    print(queue_list)

def main():
    words = ["hello", "world", "test", "Frank"]
    new_queue = list_to_queue(words)
    print('New queue: ')
    print_queue(new_queue)
    print('APPROXIMATIVE size of the queue: ', new_queue.qsize())
    print('Is it queue empty ? :', new_queue.empty())
    new_list = queue_to_list(new_queue)
    print('New list: ', new_list)
    print('Is it queue empty ? :', new_queue.empty())

if __name__ == "__main__":
    main()
