import ffmpeg
import os

# Examples: https://github.com/kkroening/ffmpeg-python/tree/master/examples
# Define the directory path
directory = ""

def extract_subs_from_mkv(file_path, stream=0):
    output_file = os.path.splitext(file_path)[0] + ".srt"
    out, _ = (
        ffmpeg
        .input(file_path)
        .output(output_file, map=f"0:s:{stream}")
        .run(capture_stdout=True)
    )
    print(out)
    return out

# Iterate through the files in the directory
for filename in os.listdir(directory):
    # Get the full path of the file
    file_path = os.path.join(directory, filename)
    
    # Check if the file is a regular file (not a directory)
    if os.path.isfile(file_path):
        # Extract subtitles
        if filename.endswith('.mkv'):
            extract_subs_from_mkv(file_path)

