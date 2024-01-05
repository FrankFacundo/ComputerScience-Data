import time

from fastapi import FastAPI

app = FastAPI()


def wait_operation():
    time.sleep(5)


@app.get("/")
async def read_root():
    wait_operation()
    return {"message": "Hello World"}


# Run the app with Uvicorn if this script is run directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)

# In this code if making two request at the same time,
# one will be finished in about 5 seconds and the other in about 10 seconds.
# because it use only one thread.
