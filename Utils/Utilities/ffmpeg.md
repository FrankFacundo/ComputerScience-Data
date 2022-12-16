# FFMPEG

## Merge mp4 video with subtitle srt file

The format MP4 does not allow to personalize (color, etc) subtitles. If wanted, use MKV instead.

```shell
ffmpeg -i $video_input -i $subtitle -c copy -c:s mov_text $video_output
```

## Merge mkv video with srt file

```shell
ffmpeg -i $video_input -i $subtitle -map 0 -map 1 -c copy -metadata:s:s:1 title="title" $video_output
```

## Extract audio from video
```shell
ffmpeg -i input-video.avi -vn -acodec copy output-audio.aac
```

## Convert avi to mp4
```shell
ffmpeg -i input_filename.avi -c:v copy -c:a copy -y output_filename.mp4
```

## Convert aac to mp3
```shell
ffmpeg -i output-audio.aac -c:a libmp3lame -ac 2 -q:a 2 outputfile.mp3
```

## Convert m4a to mp3

```shell
ffmpeg -i Chapter19.m4a -c:v copy -c:a libmp3lame -q:a 4 Chapter19.mp3
```

## Extra package

```shell
mediainfo video.mp4
```

## Merge audio with image

H.265 State-of-the-art as of 2022/12

```shell
ffmpeg -r 1 -loop 1 -y -i black.jpg -i 21-FlashFlash.aac -c:a copy -r 1 -vcodec libx265 -shortest out.mp4
```

## Merge audio with image and subtitle

H.265 State-of-the-art as of 2022/12

```shell
ffmpeg -r 1 -loop 1 -y -i black.jpg -i 21-FlashFlash.aac -i 21.srt -c:s mov_text -c:a copy -r 1 -vcodec libx265 -shortest out.mp4
ffmpeg -loop 1 -i img.jpg -i music.m4a -shortest -acodec copy -vcodec libx265 out.mkv
```

## Convert aax to mp3

To get activation_bytes of Audible check https://github.com/inAudible-NG/tables

```shell
ffmpeg -y -activation_bytes $AB -i book.aax -c:a copy -vn book.m4a
```
