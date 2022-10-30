# FFMPEG

## Merge video with subtitle srt file

```shell
ffmpeg -i $video_input -i $subtitle -c copy -c:s mov_text $video_output
```

## Extract srt from mkv file

```shell
ffmpeg -i $video_input -i $subtitle -map 0 -map 1 -c copy -metadata:s:s:1 title="title" $video_output
```
