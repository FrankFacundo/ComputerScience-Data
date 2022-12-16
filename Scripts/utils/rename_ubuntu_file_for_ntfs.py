import os
import re
from pathlib import Path


def rename():
    files = Path("/home/user/Videos").glob('*')
    for file in files:
        new_name = re.sub(":", "-", str(file))
        print(f'Filename: {file}')
        print(f'New name filename: {new_name}')
        os.rename(str(file), new_name)

rename()
