#! /usr/bin/env python3
# To add more features check https://github.com/plutowang/generate-video-subtitle
import json
import os
import timestr

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
        print(str_segment)
        result_srt += str_segment
    
    with open(output_path + 'transcript-subtitle.srt', 'w') as f:
        f.write(result_srt)
    print(result_srt)
    return


def main():
    # arg = sys.argv[1]
    output_path = './output/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    whisper_result = "whisper_result.json"
    with open(whisper_result, 'r', encoding='utf8') as f:
        response = json.load(f)

    # write into subtitle
    try:
        write_into_subtitle(response, output_path)
        print("Write into subtitle successfully!")
    except BaseException as e:
        print(e)
        print('error: Write into subtitle failed!')
        exit(1)


if __name__ == "__main__":
    main()
