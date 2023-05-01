from pytube import YouTube
from pytube import Playlist
from slugify import slugify
import ffmpeg
import csv


def DownloadVideo(video_link,folder,maxres=None):
    if maxres==None:
        print("Video Started")
        video_file = YouTube(video_link).streams.order_by('resolution').desc().first().download()
        print("Video Done")
    else:
        print("Video Started")
        video_file = YouTube(video_link).streams.filter(res=maxres).order_by('resolution').desc().first().download()
        print("Video Done")
        
    # print(video_file)
    video_name = slugify(video_file.replace(".webm","").split("/")[-1])
    print("Audio Started")
    audio_file = YouTube(video_link).streams.filter(only_audio=True).order_by('abr').desc().first().download(filename_prefix="audio_")
    print("Audio Done")
    source_audio = ffmpeg.input(audio_file)
    source_video = ffmpeg.input(video_file)
    print("Concatenation Started")
    ffmpeg.concat(source_video, source_audio, v=1, a=1).output(f"{folder}/{video_name}.mp4").run()
    print("Concatenation Done")
    return None
        

def DownloadChannel(channel_link,folder,maxres=None):
    pure_link = channel_link.replace("/featured","/videos")
    list_videos = Playlist(pure_link).video_urls
    video_count = 0
    total_video = len(list_videos)
    print(f'{total_video} Videos Found')
    list_videos_downloaded = []
    with open('youtube_export_history.csv', 'r', newline='') as csvfile:
        spamwriter = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)
        for row in spamwriter:
            list_videos_downloaded.append(row[0])
            
    for video in list_videos:
        if video in list_videos_downloaded:
            video_count = video_count + 1
            print(f'Video {video_count}/{total_video} already downloaded')
        else:
            print(video)
            video_count = video_count + 1
            print(f'{video_count}/{total_video} Started')
            DownloadVideo(video_link=video,maxres=maxres,folder=folder)

            with open('youtube_export_history.csv', 'a', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([video])

            print(f'{video_count}/{total_video} Done')
            
DownloadChannel(channel_link="channel_link",
                folder="fullFolderPath",
                maxres=None)