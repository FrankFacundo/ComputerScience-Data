from glob import glob
import os
import re
import subprocess

DIR_PATH = os.getenv('VIDEO_DIR')
SUBTITLE_EXTENSION = '.srt'
SUBTITLE_SUFFIX = ' (merged)'
VIDEO_OUTPUT_SUFFIX = '_double_sub.mp4'

def convert_string_to_command(text: str) -> str:
    space_leftparenthesis_rightparenthesis_regex = r'(\s|\(|\))'
    bachslash_with_capture_regex = r'\\\1'
    return re.sub(space_leftparenthesis_rightparenthesis_regex, bachslash_with_capture_regex, text)

def merge_video_with_subtitle():
    for file in glob(DIR_PATH + "/*.mp4"):
        print('*************')
        video_input = file
        print('Video input : {}'.format(video_input))
        size_dot_and_extension = 4
        video_input_without_extension = video_input[:-size_dot_and_extension]
        subtitle = video_input_without_extension + SUBTITLE_SUFFIX + SUBTITLE_EXTENSION
        print('Subtitle file : {}'.format(subtitle))
        video_output = video_input_without_extension + VIDEO_OUTPUT_SUFFIX
        print('Video output : {}'.format(video_output))

        command = "ffmpeg -i {} -i {} -c copy -c:s mov_text {}".format(
            convert_string_to_command(video_input),
            convert_string_to_command(subtitle),
            convert_string_to_command(video_output))

        print('command : {}'.format(command))
        # Why subprocess and not os.system ?
        # https://stackoverflow.com/questions/89228/how-do-i-execute-a-program-or-call-a-system-command
        subprocess.call(command, shell=True)


merge_video_with_subtitle()

