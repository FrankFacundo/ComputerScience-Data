import os
import subprocess

from glob import glob


def extract_audio_from_video(filepath, output_name=None):
    command = f'mediainfo --Inform="Audio;%Format%" {filepath}'
    output, _ = exec_command(command, shell=True)
    audio_ext = output[:-1].lower()

    if output_name is None:
        output_name = f"{os.path.basename(filepath)[:-4]}.{audio_ext}"

    command = f"ffmpeg -i {filepath} -vn -acodec copy {output_name}"
    exec_command(command, shell=True)
    return output_name

def exec_command(command, shell=False):
    if shell:
        print(f"command to execute: {command}")
    else:
        print(f"command to execute: {' '.join(command)}")
    proc = subprocess.Popen(command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, shell=shell)
    output, error = proc.communicate()

    return output.decode("utf-8"), error.decode("utf-8")

def convert_to_mp3(filepath, output_name=None):
    if output_name is None:
        SIZE_EXTENSION = len(".aac")
        output_name = f"{os.path.basename(filepath)[:-SIZE_EXTENSION]}.mp3"

    command = f"ffmpeg -i {filepath} -c:a libmp3lame -ac 2 -q:a 2 {output_name}"
    exec_command(command, shell=True)

def extract_mp3(filepath, output_name=None):
    path_audio_extracted = extract_audio_from_video(filepath, output_name)
    convert_to_mp3(filepath, output_name)
    exec_command(["rm", "-rf", path_audio_extracted])

def main(directory):
    for file in glob(directory + "/*.mp4"):
        print(file)


directory = ""
main(directory)