# FFMPEG

## Merge mp4 video with subtitle srt file

```shell
ffmpeg -i $video_input -i $subtitle -c copy -c:s mov_text $video_output
```

## Extract srt from mkv file

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

## Extra package

```shell
mediainfo video.mp4
```
