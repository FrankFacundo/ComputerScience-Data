import os
import re
import subprocess

# Define the directory containing the mkv video files
video_directory = "/path/video/directory/input"  # Replace with the path to your video directory
res_directory = (
    "/path/output"  # Replace with the path to your video directory
)


def convert_string_to_command(text: str) -> str:
    space_leftparenthesis_rightparenthesis_regex = r"(\s|\(|\)|\')"
    bachslash_with_capture_regex = r"\\\1"
    return re.sub(
        space_leftparenthesis_rightparenthesis_regex, bachslash_with_capture_regex, text
    )


# video_filename = []
for filename in os.listdir(video_directory):
    if filename.endswith(".mkv"):
        print(filename)
        sub_filename = filename[:-4] + ".str"
        subprocess.call(command, shell=True)
