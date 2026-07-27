from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from mutagen.mp3 import MP3

def create_video(photo_paths, audio_paths, output_path="output/final_video.mp4"):
    clips = []
    for photo, audio in zip(photo_paths, audio_paths):
        duration = MP3(audio).info.length
        img_clip = ImageClip(photo).set_duration(duration)
        audio_clip = AudioFileClip(audio)
        video_clip = img_clip.set_audio(audio_clip)
        clips.append(video_clip)

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(output_path, fps=24)
    return output_path