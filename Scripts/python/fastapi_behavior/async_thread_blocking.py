import asyncio
import time

from fastapi import FastAPI

app = FastAPI()


def wait_operation():
    time.sleep(5)


@app.get("/")
async def read_root():
    loop = asyncio.get_running_loop()
    # Run the blocking operation in a separate thread
    await loop.run_in_executor(None, wait_operation)
    return {"message": "Hello World"}


# In this code if making two request at the same time,
# both will be finished in about 5 seconds because it uses two executors (two threads).
