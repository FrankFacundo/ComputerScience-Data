import shutil
import os
import re

from typing import Dict
from dataclasses import dataclass
from pprint import pprint

import utils

@dataclass
class VideoMetadata:
    key: str
    video_directory : str
    subtitle_directory: str = ""
    original_video_name : str = ""
    original_subtitle_name : str = ""
    new_subtitle_name : str = ""

class SubtitleUtils(object):

    mapping: Dict[str, VideoMetadata] = {}
    translation_dir = "translations"
    translation_filename_suffix = "_merged"

    def add_subs(self, video_dir, subtitle_dir=None, source_language=None, destination_language=None, verbose=False):
        if subtitle_dir is None:
            subtitle_dir = video_dir
        if destination_language is not None and source_language is not None:
            # Create a directory inside subtitle_dir
            subtitles_translated_dir = os.path.join(subtitle_dir, SubtitleUtils.translation_dir)
            if not os.path.exists(subtitles_translated_dir):
                os.makedirs(subtitles_translated_dir)

            subtitle_files = [file for file in os.listdir(subtitle_dir) if os.path.isfile(os.path.join(subtitle_dir, file))]
            for original_subtitle_file in subtitle_files:
                if original_subtitle_file.endswith(".srt"):
                    original_subtitle_path = os.path.join(subtitle_dir, original_subtitle_file)
                    print(original_subtitle_path)
                    translation_filename = os.path.splitext(original_subtitle_file)[0] + SubtitleUtils.translation_filename_suffix + ".srt"
                    translation_path = os.path.join(subtitles_translated_dir, translation_filename)
                    if not os.path.exists(translation_path):
                        self.translate_subs(subtitle_path=original_subtitle_path,
                                            source_language=source_language,
                                            destination_language=destination_language,
                                            verbose=verbose)

            subtitle_dir = subtitles_translated_dir
        self.map_video_subtitles(video_dir, subtitle_dir=subtitle_dir)

        for video_metadata in self.mapping.values():
            original_subtitle_path = os.path.join(subtitle_dir, video_metadata.original_subtitle_name)
            new_subtitle_path = os.path.join(video_dir, video_metadata.new_subtitle_name)
            shutil.copy(original_subtitle_path, new_subtitle_path)

    # best it to translate over threads.
    @staticmethod
    def translate_subs(subtitle_path, source_language='en', destination_language='es', color_left='#66ffff', verbose=False):
        subtitle_dir = os.path.dirname(subtitle_path)
        subtitles_translated_dir = os.path.join(subtitle_dir, SubtitleUtils.translation_dir)

        command = f"java -jar dual_sub.jar \
            --input '{SubtitleUtils.string_to_shell_command(subtitle_path)}' \
            --output '{subtitles_translated_dir}' \
            --sourceLanguage '{source_language}' \
            --destinationLanguage '{destination_language}' \
            --colorLeft '{color_left}' \
            "
        print("command: ", command)
        utils.exec_command(command, shell=True, verbose=verbose)
        return


    @staticmethod
    def string_to_shell_command(string: str):
        return string.replace("'", "'\\''")

    def map_video_subtitles(self,video_dir, subtitle_dir=None):
        self.map_video_name(video_dir)
        self.map_subtitle_name(subtitle_dir)


    def map_subtitle_name(self, directory, lang='eng'):
        print("\n================================")
        subtitle_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for original_subtitle_file in subtitle_files:
            if original_subtitle_file.endswith(".srt"):
                is_chapter_season_video = re.search(r's\d{2}e\d{2}', original_subtitle_file, re.IGNORECASE)
                chapter_season_code = is_chapter_season_video.group()
                print(chapter_season_code)
                if is_chapter_season_video and chapter_season_code in self.mapping:
                    video_metadata = self.mapping[chapter_season_code]
                    video_name_without_ext = os.path.splitext(video_metadata.original_video_name)[0]
                    new_subtitle_name = video_name_without_ext + "." + lang + ".srt"
                    video_metadata.original_subtitle_name = original_subtitle_file
                    video_metadata.new_subtitle_name = new_subtitle_name
                    video_metadata.subtitle_directory = directory
        pprint(self.mapping)
        return chapter_season_code

    def map_video_name(self, directory):

        # print("\n================================")
        video_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for video_file in video_files:
            if video_file.endswith((".mp4", ".mkv", ".avi", ".mpeg", ".mpg")):
                is_chapter_season_video = re.search(r's\d{2}e\d{2}', video_file, re.IGNORECASE)
                chapter_season_code = is_chapter_season_video.group()
                # print(chapter_season_code)
                if is_chapter_season_video:
                    video_metadata : VideoMetadata = VideoMetadata(key=chapter_season_code, 
                                                                   video_directory=directory, 
                                                                   original_video_name=video_file)
                    self.mapping[chapter_season_code] = video_metadata
        # pprint(self.mapping)
        return chapter_season_code



path_video = "./subtitles"
path_subs = "./subtitles/english_subtitles2"
subtitle_utils = SubtitleUtils()
subtitle_utils.add_subs(video_dir=path_video, 
                        subtitle_dir=path_subs, 
                        source_language="en", 
                        destination_language="es", 
                        verbose=True)
