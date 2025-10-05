from openai import OpenAI

client = OpenAI()
audio_file = open("the_big_short.mp3", "rb")

transcription = client.audio.transcriptions.create(
    file=audio_file,
    model="whisper-1",
    response_format="verbose_json",
    timestamp_granularities=["word"],
)

print(transcription.words)
