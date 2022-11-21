from glob import glob
import os
import re
import subprocess

DIR_PATH = os.getenv('VIDEO_DIR')
SUBTITLE_EXTENSION = '.srt'
SUBTITLE_SUFFIX = ' (merged)'
VIDEO_OUTPUT_SUFFIX = '_double_sub.'

FORMAT_MP4 = "mp4"

def convert_string_to_command(text: str) -> str:
    space_leftparenthesis_rightparenthesis_regex = r'(\s|\(|\)|\')'
    bachslash_with_capture_regex = r'\\\1'
    return re.sub(space_leftparenthesis_rightparenthesis_regex, bachslash_with_capture_regex, text)

def merge_video_with_subtitle(video_extension = FORMAT_MP4, languages = "EN:ES"):
    print('DIR_PATH : {}'.format(DIR_PATH))
    if DIR_PATH is None:
        print("None DIR_PATH, please declare the environment variable.")
        raise Exception
    else:
        for file in glob(DIR_PATH + "/*." + video_extension):
            print('*************')
            subtitle_code = ""
            video_input = file
            print('Video input : {}'.format(video_input))
            size_dot_and_extension = 4
            video_input_without_extension = video_input[:-size_dot_and_extension]
            subtitle = video_input_without_extension + SUBTITLE_SUFFIX + SUBTITLE_EXTENSION
            print('Subtitle file : {}'.format(subtitle))
            video_output = video_input_without_extension + VIDEO_OUTPUT_SUFFIX + video_extension
            print('Video output : {}'.format(video_output))

            if video_extension == FORMAT_MP4:
                params = "-c copy -c:s mov_text"
            elif video_extension == "mkv":
                params = '-map 0 -map 1 -c copy -metadata:s:s:1 title="{}"'.format(languages)
            else:
                print("Unknown format to add subtitles")
                raise Exception
            # Check documentation for subtitles:
            # https://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/subtitle_options
            command = "ffmpeg -i {} -i {} {} {}".format(
                convert_string_to_command(video_input),
                convert_string_to_command(subtitle),
                params,
                convert_string_to_command(video_output))

            print('Command : {}'.format(command))
            # Why subprocess and not os.system ?
            # https://stackoverflow.com/questions/89228/how-do-i-execute-a-program-or-call-a-system-command
            subprocess.call(command, shell=True)

def extract_subtitle_from_video():
    print('DIR_PATH : {}'.format(DIR_PATH))
    if DIR_PATH is None:
        print("None DIR_PATH, please declare the environment variable.")
        raise Exception
    else:
        for file in glob(DIR_PATH + "/*.mkv"):
            print('*************')
            video_input = file
            print('Video input : {}'.format(video_input))
            size_dot_and_extension = 4
            video_input_without_extension = video_input[:-size_dot_and_extension]
            subtitle = video_input_without_extension + SUBTITLE_EXTENSION
            print('Subtitle file : {}'.format(subtitle))

            command = "ffmpeg -i {} -map 0:s:0 {}".format(
                convert_string_to_command(video_input),
                convert_string_to_command(subtitle))

            print('command : {}'.format(command))
            # Why subprocess and not os.system ?
            # https://stackoverflow.com/questions/89228/how-do-i-execute-a-program-or-call-a-system-command
            subprocess.call(command, shell=True)

 
merge_video_with_subtitle(video_extension = "mkv", languages="English:Spanish")

# extract_subtitle_from_video()