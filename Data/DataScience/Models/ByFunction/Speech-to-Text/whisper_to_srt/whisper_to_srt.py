#! /usr/bin/env python3
# To add more features check https://github.com/plutowang/generate-video-subtitle
import json
import os
import timestr
from glob import glob

def write_into_subtitle(response, output_path):
    result_srt = ""
    i = 0
    for segment in response["segments"]:
        str_segment = ""
        str_segment += str(i)
        str_segment += '\n'
        str_segment += timestr.timefm(segment["start"])
        str_segment += ' --> '
        str_segment += timestr.timefm(segment["end"])
        str_segment += '\n'
        str_segment += segment["text"]
        str_segment += '\n\n'
        i+=1
        # print(str_segment)
        result_srt += str_segment
    
    with open(output_path, 'w') as f:
        f.write(result_srt)
    print(result_srt)
    return


def main(directory):
    # arg = sys.argv[1]
    output_path = './output/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in glob(directory + "/*.json"):
        print("file", file)
        path_output = file[:-5] + ".srt"
        print("path_output", path_output)
        response = None
        whisper_result = file
        with open(whisper_result, 'r', encoding='utf8') as f:
            response = json.load(f)

        # write into subtitle
        try:
            write_into_subtitle(response, path_output)
            print("Write into subtitle successfully!")
        except BaseException as e:
            print(e)
            print('error: Write into subtitle failed!')
            exit(1)


if __name__ == "__main__":
    directory = "video"
    main(directory)

