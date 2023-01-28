# Doc: https://instaloader.github.io/module/structures.html#posts
from instaloader import Instaloader, Profile

L = Instaloader(
        download_pictures=False,
        download_video_thumbnails=False,
        download_videos=True,
        save_metadata=False,
        dirname_pattern=None,
        filename_pattern=None,
        title_pattern=None,
        sanitize_paths=False
    )

PROFILE = "filmmakersworld"
profile = Profile.from_username(L.context, PROFILE)

posts= profile.get_posts()

for post in posts:
    print(post)
    L.download_post(post, PROFILE)
