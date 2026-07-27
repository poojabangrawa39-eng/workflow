
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Tuple 

import imageio_ffmpeg
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
from moviepy.editor import VideoClip, concatenate_videoclips
from PIL import Image, ImageOps


# =========================================================
# MAIN SETTINGS
# =========================================================
PHOTO_DURATION = 5.5       # Each photo stays between 5 and 6 seconds
ZOOM_AMOUNT = 0.08         # Smooth 8% movement with the complete photo always visible
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
VIDEO_FPS = 30             # Smoother motion than 24 FPS
AUDIO_SAMPLE_RATE = 44100
MIN_VOICE_SPEED = 1.08     # Natural voice with a small speed increase
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
APP_VERSION = "Generate script first then choose AI or uploaded voice v15"


# =========================================================
# STREAMLIT SETUP
# =========================================================
st.set_page_config(
    page_title="Real Estate Video Generator",
    page_icon="🏠",
    layout="wide",
)
load_dotenv()


# =========================================================
# BASIC HELPERS
# =========================================================
def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key

    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        return ""


def clean_area_name(name: str) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"\.(jpg|jpeg|png|webp)$", "", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value or "flat interior"


def prepare_photo_labels(sequence_text: str, total_photos: int) -> List[str]:
    labels = [
        clean_area_name(line)
        for line in sequence_text.splitlines()
        if line.strip()
    ]

    while len(labels) < total_photos:
        labels.append(f"flat interior {len(labels) + 1}")

    return labels[:total_photos]


def count_spoken_words(text: str) -> int:
    words = re.findall(r"[A-Za-z0-9\u0900-\u097F]+", text)
    return len(words)


def prepare_custom_voiceovers(
    custom_text: str,
    total_photos: int,
) -> List[str]:
    """Read one custom voiceover line for each photo."""
    lines = []

    for line in custom_text.splitlines():
        sentence = re.sub(r"\s+", " ", line).strip()
        if sentence:
            lines.append(sentence)

    if len(lines) != total_photos:
        raise ValueError(
            f"Enter exactly {total_photos} voiceover lines because you have "
            f"{total_photos} photos. Currently you entered {len(lines)} lines."
        )

    return lines


def create_file_token(index: int, uploaded_file) -> str:
    """Create a unique stable ID for every uploaded file."""
    size = getattr(uploaded_file, "size", 0)
    return f"{index}::{uploaded_file.name}::{size}"


def ensure_exact_photo_order_state(uploaded_files) -> List[str]:
    """Keep the user-selected photo sequence stable across reruns."""
    current_tokens = [
        create_file_token(index, uploaded_file)
        for index, uploaded_file in enumerate(uploaded_files)
    ]
    signature = "||".join(current_tokens)

    if st.session_state.get("photo_order_signature") != signature:
        st.session_state["photo_order_signature"] = signature
        st.session_state["photo_order_tokens"] = current_tokens.copy()

    saved_tokens = st.session_state.get("photo_order_tokens", []).copy()

    # Remove deleted files and append any new files while preserving
    # the existing chosen order for files that are still present.
    saved_tokens = [token for token in saved_tokens if token in current_tokens]
    for token in current_tokens:
        if token not in saved_tokens:
            saved_tokens.append(token)

    st.session_state["photo_order_tokens"] = saved_tokens
    return saved_tokens


def move_photo_in_order(position: int, direction: int) -> None:
    """Move one photo up or down in the exact order list."""
    tokens = st.session_state.get("photo_order_tokens", []).copy()
    new_position = position + direction

    if 0 <= position < len(tokens) and 0 <= new_position < len(tokens):
        tokens[position], tokens[new_position] = tokens[new_position], tokens[position]
        st.session_state["photo_order_tokens"] = tokens


# =========================================================
# VOICEOVER GENERATION
# =========================================================
def generate_voiceovers(
    client: Groq,
    labels: List[str],
    property_details: str,
) -> List[str]:
    photo_list = "\n".join(
        f"Photo {index}: {label}"
        for index, label in enumerate(labels, start=1)
    )

    prompt = f"""
Create Hindi-Hinglish real-estate voiceover lines for a photo video.

PROPERTY DETAILS:
{property_details.strip()}

EXACT PHOTO ORDER:
{photo_list}

STRICT RULES:
1. Return exactly {len(labels)} lines in the same order as the photos.
2. Every line must describe only its own photo area.
3. Each line must contain 14 to 16 spoken words so it naturally fills
   about {PHOTO_DURATION} seconds at a normal speaking speed.
4. Do not mention the next photo in the current photo line.
5. Mention the location only in Photo 1.
6. Mention Jaipur Rental or contact only in the final photo.
7. Use simple professional Hindi-Hinglish.
8. Keep every line different and avoid repeated sentence structures.
9. Do not invent facilities that are not in the property details.
10. Make sure use only hinglish language , follow this rule strictly
11. Avoid commas, semicolons, dashes and extra punctuation because they
    create voice pauses.
12. Return only valid JSON in exactly this format:
{{"voiceovers": ["line 1", "line 2", "line 3"]}}
""".strip()

    last_error = "Voiceover generation failed."

    for attempt in range(4):
        retry_note = ""
        if attempt:
            retry_note = (
                "\nRegenerate all lines. Keep every line strictly between "
                "14 and 16 spoken words and keep the exact photo order."
            )

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write short photo-synced Hindi-Hinglish "
                            "real-estate voiceovers and return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt + retry_note},
                ],
                temperature=0.75,
                max_tokens=1200,
            )

            content = (response.choices[0].message.content or "").strip()
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.I)
            content = re.sub(r"\s*```$", "", content)

            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("AI did not return JSON.")

            data = json.loads(content[start : end + 1])
            lines = data.get("voiceovers", [])

            if not isinstance(lines, list) or len(lines) != len(labels):
                raise ValueError(
                    f"Expected {len(labels)} lines but received {len(lines)}."
                )

            cleaned: List[str] = []
            invalid_lengths = []

            for index, line in enumerate(lines, start=1):
                sentence = re.sub(r"[,;:।.!?\n]+", " ", str(line))
                sentence = re.sub(r"\s+", " ", sentence).strip()
                sentence = sentence.strip("-— ")

                if not sentence:
                    raise ValueError(f"Photo {index} voiceover is empty.")

                word_count = count_spoken_words(sentence)
                if word_count < 12 or word_count > 18:
                    invalid_lengths.append((index, word_count))

                cleaned.append(sentence)

            # A small tolerance keeps generation reliable. Audio is still
            # fitted exactly to the photo slot later with FFmpeg.
            if invalid_lengths and attempt < 3:
                raise ValueError(
                    "Voiceover word counts were outside the safe range: "
                    + str(invalid_lengths)
                )

            return cleaned

        except Exception as error:
            last_error = str(error)

    raise RuntimeError(last_error)


# =========================================================
# PHOTO ZOOM VIDEO
# =========================================================
def make_zoom_clip(
    image: Image.Image,
    index: int,
    photo_duration: float,
) -> VideoClip:
    """Create a stable zoom effect with a black background and no cropping.

    The complete photo stays visible. A floating-point affine transform is
    used instead of resizing to changing integer dimensions, which prevents
    the one-pixel shaking that can happen during frame-by-frame resizing.
    """
    resampling = getattr(Image, "Resampling", Image)
    transform_mode = getattr(Image, "Transform", Image).AFFINE

    source_image = image.convert("RGB")

    # Fit the full photo inside the vertical frame without cropping.
    # The 96% safety margin ensures the largest zoom frame still has space.
    maximum_box = (
        max(1, int(VIDEO_WIDTH * 0.96)),
        max(1, int(VIDEO_HEIGHT * 0.96)),
    )
    complete_photo = ImageOps.contain(
        source_image,
        maximum_box,
        method=resampling.LANCZOS,
    )

    # Place the complete photo once on a transparent fixed-size canvas.
    # The canvas size never changes, which removes resize-related shaking.
    foreground = Image.new(
        "RGBA",
        (VIDEO_WIDTH, VIDEO_HEIGHT),
        (0, 0, 0, 0),
    )
    paste_x = (VIDEO_WIDTH - complete_photo.width) // 2
    paste_y = (VIDEO_HEIGHT - complete_photo.height) // 2
    foreground.paste(
        complete_photo.convert("RGBA"),
        (paste_x, paste_y),
    )

    black_background = Image.new(
        "RGBA",
        (VIDEO_WIDTH, VIDEO_HEIGHT),
        (0, 0, 0, 255),
    )

    center_x = VIDEO_WIDTH / 2.0
    center_y = VIDEO_HEIGHT / 2.0
    minimum_scale = 1.0 - ZOOM_AMOUNT

    # Photo 1 zooms in, Photo 2 zooms out, then repeats.
    zoom_in = index % 2 == 0

    def make_frame(t: float) -> np.ndarray:
        progress = min(max(t / photo_duration, 0.0), 1.0)

        # Smoothstep easing gives zero movement speed at both ends.
        eased = progress * progress * (3.0 - (2.0 * progress))

        if zoom_in:
            scale = minimum_scale + ((1.0 - minimum_scale) * eased)
        else:
            scale = 1.0 - ((1.0 - minimum_scale) * eased)

        inverse_scale = 1.0 / scale
        offset_x = center_x * (1.0 - inverse_scale)
        offset_y = center_y * (1.0 - inverse_scale)

        # Floating-point affine coordinates keep the motion centred and stable.
        transformed_foreground = foreground.transform(
            (VIDEO_WIDTH, VIDEO_HEIGHT),
            transform_mode,
            (
                inverse_scale, 0.0, offset_x,
                0.0, inverse_scale, offset_y,
            ),
            resample=resampling.BICUBIC,
        )

        frame = Image.alpha_composite(
            black_background,
            transformed_foreground,
        ).convert("RGB")

        return np.asarray(frame, dtype=np.uint8)

    return VideoClip(make_frame=make_frame, duration=photo_duration)


def load_images(uploaded_files) -> List[Image.Image]:
    images = []

    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        with Image.open(uploaded_file) as image:
            images.append(image.convert("RGB").copy())

    return images


# =========================================================
# AUDIO SYNC HELPERS
# =========================================================
def ffprobe_duration(file_path: Path, ffmpeg_exe: str) -> float:
    """Read media duration using ffprobe next to imageio-ffmpeg's ffmpeg."""
    ffmpeg_path = Path(ffmpeg_exe)
    ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    ffprobe_path = ffmpeg_path.with_name(ffprobe_name)

    if ffprobe_path.exists():
        command = [
            str(ffprobe_path),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass

    # Fallback: ask ffmpeg to print media information and parse Duration.
    result = subprocess.run(
        [ffmpeg_exe, "-i", str(file_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr,
    )
    if not match:
        raise RuntimeError(f"Could not read duration of {file_path.name}.")

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return (hours * 3600) + (minutes * 60) + seconds


def get_atempo_filter(speed_factor: float) -> str:
    """Create an FFmpeg atempo chain for any positive speed factor."""
    speed_factor = max(float(speed_factor), 0.01)
    filters: List[str] = []

    while speed_factor > 2.0:
        filters.append("atempo=2.0")
        speed_factor /= 2.0

    while speed_factor < 0.5:
        filters.append("atempo=0.5")
        speed_factor /= 0.5

    filters.append(f"atempo={speed_factor:.6f}")
    return ",".join(filters)


def create_raw_tts(text: str, output_path: Path) -> None:
    tts = gTTS(
        text=text,
        lang="hi",
        slow=False,
        tld="co.in",
    )
    tts.save(str(output_path))


def trim_tts_audio(
    raw_path: Path,
    trimmed_path: Path,
    ffmpeg_exe: str,
) -> None:
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(raw_path),
        "-af",
        (
            "silenceremove="
            "start_periods=1:start_duration=0.02:start_threshold=-45dB:"
            "stop_periods=-1:stop_duration=0.04:stop_threshold=-45dB"
        ),
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(trimmed_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not trim voice audio.\n" + result.stderr[-1200:]
        )


def prepare_continuous_audio_segment(
    trimmed_path: Path,
    segment_path: Path,
    preferred_duration: float,
    ffmpeg_exe: str,
) -> Tuple[float, float]:
    """Prepare one voice line without adding silence after it.

    The voice remains close to normal speed. Each photo's duration is later
    matched to the real duration of its own voice line, so the next line and
    the next photo start together with no silent break.
    """
    raw_duration = ffprobe_duration(trimmed_path, ffmpeg_exe)

    # Keep the voice natural. Only make a moderate speed adjustment toward
    # the preferred photo duration. No apad is used, so no silent tail exists.
    required_speed = raw_duration / max(preferred_duration, 0.5)
    speed_factor = min(max(required_speed, 0.92), 1.18)
    atempo_filter = get_atempo_filter(speed_factor)

    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(trimmed_path),
        "-af",
        f"{atempo_filter},asetpts=N/SR/TB",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        str(segment_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not prepare continuous voice audio.\n"
            + result.stderr[-1200:]
        )

    actual_duration = ffprobe_duration(segment_path, ffmpeg_exe)
    return speed_factor, actual_duration


def create_synced_audio(
    voiceovers: List[str],
    temp_path: Path,
    preferred_duration: float,
    ffmpeg_exe: str,
) -> Tuple[Path, List[float], List[float]]:
    """Create one continuous narration in the exact photo order.

    No silence is padded between voice lines. Segment durations are returned
    so every photo can stay visible for exactly its own spoken line.
    """
    segment_paths: List[Path] = []
    speed_factors: List[float] = []
    segment_durations: List[float] = []

    for index, line in enumerate(voiceovers, start=1):
        raw_path = temp_path / f"voice_raw_{index:02d}.mp3"
        trimmed_path = temp_path / f"voice_trimmed_{index:02d}.wav"
        segment_path = temp_path / f"voice_segment_{index:02d}.wav"

        create_raw_tts(line, raw_path)
        trim_tts_audio(raw_path, trimmed_path, ffmpeg_exe)

        speed_factor, actual_duration = prepare_continuous_audio_segment(
            trimmed_path,
            segment_path,
            preferred_duration,
            ffmpeg_exe,
        )

        segment_paths.append(segment_path)
        speed_factors.append(speed_factor)
        segment_durations.append(actual_duration)

    # FFmpeg concat joins the WAV samples directly in the same order.
    # No extra silent file or delay is inserted.
    concat_file = temp_path / "audio_segments.txt"
    concat_lines = []

    for segment_path in segment_paths:
        safe_path = segment_path.resolve().as_posix().replace("'", "'\\''")
        concat_lines.append(f"file '{safe_path}'")

    concat_file.write_text("\n".join(concat_lines), encoding="utf-8")
    combined_audio_path = temp_path / "continuous_voice.wav"

    command = [
        ffmpeg_exe,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-ac",
        "2",
        str(combined_audio_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not join continuous voice segments.\n"
            + result.stderr[-1500:]
        )

    return combined_audio_path, speed_factors, segment_durations


# =========================================================
# FINAL VIDEO CREATION
# =========================================================
def safe_remove_folder(folder_path: str) -> None:
    for _ in range(8):
        try:
            shutil.rmtree(folder_path, ignore_errors=False)
            return
        except PermissionError:
            time.sleep(0.4)
        except FileNotFoundError:
            return

    shutil.rmtree(folder_path, ignore_errors=True)


def create_video_bytes(
    images: List[Image.Image],
    voiceovers: List[str],
    preferred_photo_duration: float,
) -> Tuple[bytes, List[float], List[float]]:
    """Create the final video without changing photo order.

    The image list is used exactly as supplied: first uploaded photo first,
    second uploaded photo second, and so on. Each photo duration matches its
    own voice line, creating continuous narration with exact photo syncing.
    """
    temp_dir = tempfile.mkdtemp(prefix="real_estate_continuous_")
    temp_path = Path(temp_dir)

    silent_video_path = temp_path / "silent_video.mp4"
    final_video_path = temp_path / "final_video.mp4"

    clips: List[VideoClip] = []
    video = None

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # Create narration first so we know the exact duration of every line.
        (
            continuous_audio_path,
            speed_factors,
            segment_durations,
        ) = create_synced_audio(
            voiceovers,
            temp_path,
            preferred_photo_duration,
            ffmpeg_exe,
        )

        if len(images) != len(segment_durations):
            raise RuntimeError(
                "The number of photos and generated voice lines does not match."
            )

        # Do not sort, reverse or otherwise modify the image sequence.
        for index, (image, line_duration) in enumerate(
            zip(images, segment_durations)
        ):
            clips.append(
                make_zoom_clip(
                    image,
                    index,
                    max(line_duration, 0.5),
                )
            )

        video = concatenate_videoclips(clips, method="chain")
        video.write_videofile(
            str(silent_video_path),
            fps=VIDEO_FPS,
            codec="libx264",
            audio=False,
            preset="medium",
            threads=4,
            logger=None,
        )

        video.close()
        video = None

        for clip in clips:
            clip.close()
        clips.clear()

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(silent_video_path),
            "-i",
            str(continuous_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(final_video_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Could not combine the final video and continuous voice.\n"
                + result.stderr[-1500:]
            )

        return (
            final_video_path.read_bytes(),
            speed_factors,
            segment_durations,
        )

    finally:
        if video is not None:
            video.close()

        for clip in clips:
            clip.close()

        safe_remove_folder(temp_dir)



def create_video_with_uploaded_voice_clip(
    images: List[Image.Image],
    uploaded_audio,
    voiceovers: List[str],
) -> Tuple[bytes, float, List[float]]:
    """Create a video using one complete uploaded voiceover recording.

    The uploaded recording should speak the generated script in the same
    line order. Photo durations are divided according to the number of words
    in each generated script line, which gives better photo-to-voice syncing.
    """
    if not images:
        raise ValueError("Upload at least one photo.")

    if len(images) != len(voiceovers):
        raise ValueError(
            "The number of photos and voice script lines must be equal."
        )

    temp_dir = tempfile.mkdtemp(prefix="real_estate_uploaded_voice_")
    temp_path = Path(temp_dir)

    extension = Path(uploaded_audio.name).suffix.lower() or ".mp3"
    uploaded_audio_path = temp_path / f"uploaded_voiceover{extension}"
    silent_video_path = temp_path / "silent_video.mp4"
    final_video_path = temp_path / "final_video.mp4"

    clips: List[VideoClip] = []
    video = None

    try:
        uploaded_audio.seek(0)
        uploaded_audio_path.write_bytes(uploaded_audio.read())
        uploaded_audio.seek(0)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        total_audio_duration = ffprobe_duration(
            uploaded_audio_path,
            ffmpeg_exe,
        )

        if total_audio_duration <= 0:
            raise ValueError("The uploaded voiceover clip has no valid audio.")

        minimum_photo_duration = 0.50
        required_minimum = len(images) * minimum_photo_duration

        if total_audio_duration <= required_minimum:
            raise ValueError(
                "The uploaded voiceover is too short for the number of photos. "
                "Please upload a longer recording."
            )

        word_weights = [
            max(count_spoken_words(line), 1)
            for line in voiceovers
        ]
        total_weight = sum(word_weights)
        distributable_duration = total_audio_duration - required_minimum

        photo_durations = [
            minimum_photo_duration
            + (distributable_duration * weight / total_weight)
            for weight in word_weights
        ]

        # Keep the exact confirmed photo order.
        for index, (image, photo_duration) in enumerate(
            zip(images, photo_durations)
        ):
            clips.append(
                make_zoom_clip(
                    image,
                    index,
                    photo_duration,
                )
            )

        video = concatenate_videoclips(clips, method="chain")
        video.write_videofile(
            str(silent_video_path),
            fps=VIDEO_FPS,
            codec="libx264",
            audio=False,
            preset="medium",
            threads=4,
            logger=None,
        )

        video.close()
        video = None

        for clip in clips:
            clip.close()
        clips.clear()

        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(silent_video_path),
            "-i",
            str(uploaded_audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{total_audio_duration:.3f}",
            "-movflags",
            "+faststart",
            str(final_video_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Could not combine the uploaded voice clip with the video.\n"
                + result.stderr[-1500:]
            )

        return (
            final_video_path.read_bytes(),
            total_audio_duration,
            photo_durations,
        )

    finally:
        if video is not None:
            video.close()

        for clip in clips:
            clip.close()

        safe_remove_folder(temp_dir)


# =========================================================
# USER INTERFACE
# =========================================================
st.title("🏠 Real Estate Video Generator")
st.caption(
    "First upload and arrange the photos. The app creates the complete voice "
    "script. After that, choose AI voice or upload your own recorded voiceover."
)

with st.sidebar:
    st.header("Video settings")
    st.write(f"Preferred AI photo duration: **about {PHOTO_DURATION} seconds**")
    st.write("Workflow: **Photos → Script → Voice choice → Video**")
    st.write("Voice options: **AI voice or uploaded recording**")
    st.write("Zoom: **stable with full photo and black background**")
    st.write("Photo order: **manually arranged and locked**")
    st.caption(APP_VERSION)

uploaded_photos = st.file_uploader(
    "Upload all property photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

property_details = st.text_area(
    "Property description",
    value=(
        "3 BHK flat for rent in Ganesh Nagar, Mansarovar, Jaipur. "
        "Fully furnished, AC, modular kitchen and parking. Prime location. "
        "Contact Jaipur Rental today."
    ),
    height=130,
)

sequence_text = st.text_area(
    "Photo sequence — enter one area for every photo",
    value=(
        "front area\n"
        "hall\n"
        "room 1\n"
        "room 2\n"
        "room 3\n"
        "kitchen\n"
        "washroom\n"
        "flat interior"
    ),
    height=190,
    help=(
        "The first line belongs to Photo 1, the second line belongs to "
        "Photo 2, and so on."
    ),
)

ordered_photos = []
ordered_tokens = []

if uploaded_photos:
    uploaded_photos = list(uploaded_photos)
    estimated_duration = len(uploaded_photos) * PHOTO_DURATION

    st.info(
        f"{len(uploaded_photos)} photos selected. Arrange them below in the "
        f"exact sequence you want. Estimated AI video duration: "
        f"about {estimated_duration:.1f} seconds."
    )

    token_order = ensure_exact_photo_order_state(uploaded_photos)
    token_to_file = {
        create_file_token(index, uploaded_file): uploaded_file
        for index, uploaded_file in enumerate(uploaded_photos)
    }

    st.subheader("Step 1 — Set the exact photo order")
    st.caption(
        "Use the Up and Down buttons. Photo 1 appears first, Photo 2 appears "
        "second, and the generated script follows the same order."
    )

    for position, token in enumerate(token_order):
        uploaded_photo = token_to_file[token]

        row_col1, row_col2, row_col3, row_col4 = st.columns(
            [1.1, 2.8, 0.7, 0.7]
        )

        with row_col1:
            uploaded_photo.seek(0)
            st.image(uploaded_photo, use_container_width=True)

        with row_col2:
            st.markdown(f"**Photo {position + 1}**")
            st.write(uploaded_photo.name)

        with row_col3:
            if st.button(
                "⬆️",
                key=f"move_up_{token}",
                use_container_width=True,
            ):
                move_photo_in_order(position, -1)
                st.rerun()

        with row_col4:
            if st.button(
                "⬇️",
                key=f"move_down_{token}",
                use_container_width=True,
            ):
                move_photo_in_order(position, 1)
                st.rerun()

        st.divider()

    ordered_tokens = [
        token
        for token in st.session_state.get("photo_order_tokens", [])
        if token in token_to_file
    ]
    ordered_photos = [
        token_to_file[token]
        for token in ordered_tokens
    ]

    st.subheader("Step 2 — Confirm photo and sequence matching")

    sequence_lines = [
        clean_area_name(line)
        for line in sequence_text.splitlines()
        if line.strip()
    ]

    preview_columns = st.columns(4)

    for index, uploaded_photo in enumerate(ordered_photos):
        uploaded_photo.seek(0)
        label = (
            sequence_lines[index]
            if index < len(sequence_lines)
            else "missing sequence label"
        )

        with preview_columns[index % 4]:
            st.image(
                uploaded_photo,
                caption=(
                    f"Photo {index + 1}: {uploaded_photo.name}\n"
                    f"Sequence: {label}"
                ),
                use_container_width=True,
            )

    if len(sequence_lines) == len(ordered_photos):
        st.success(
            "Every photo has one matching sequence line. This exact order "
            "will be used to create the script and video."
        )
    else:
        st.error(
            f"You uploaded {len(ordered_photos)} photos but entered "
            f"{len(sequence_lines)} sequence lines. Enter exactly one "
            "sequence line for every photo."
        )

current_script_signature = (
    "||".join(ordered_tokens)
    + "##"
    + property_details.strip()
    + "##"
    + sequence_text.strip()
)

st.subheader("Step 3 — Generate the voice script")

if st.button(
    "📝 Generate Voice Script",
    type="primary",
    use_container_width=True,
):
    api_key = get_groq_api_key()
    sequence_lines = [
        line.strip()
        for line in sequence_text.splitlines()
        if line.strip()
    ]

    if not ordered_photos:
        st.error("Upload and arrange at least one photo first.")
    elif len(sequence_lines) != len(ordered_photos):
        st.error(
            "The number of sequence lines must be exactly equal to "
            "the number of photos."
        )
    elif not property_details.strip():
        st.error("Enter the property description.")
    elif not api_key:
        st.error("Add GROQ_API_KEY to your .env file first.")
    else:
        try:
            labels = prepare_photo_labels(
                sequence_text,
                len(ordered_photos),
            )
            client = Groq(api_key=api_key)

            with st.spinner(
                "Creating one matching voice script line for every photo..."
            ):
                generated_lines = generate_voiceovers(
                    client,
                    labels,
                    property_details,
                )

            st.session_state["generated_voice_script"] = "\n".join(
                generated_lines
            )
            st.session_state["voice_script_signature"] = (
                current_script_signature
            )
            st.success(
                "Voice script created. Review it below, then choose the voice option."
            )

        except Exception as error:
            st.exception(error)

saved_script = st.session_state.get("generated_voice_script", "")
saved_signature = st.session_state.get("voice_script_signature", "")
script_is_current = (
    bool(saved_script)
    and saved_signature == current_script_signature
)

if saved_script:
    st.subheader("Step 4 — Review or edit the generated script")

    if not script_is_current:
        st.warning(
            "The photos, photo order, sequence, or property description changed "
            "after this script was generated. Generate the voice script again."
        )

    edited_script = st.text_area(
        "Generated voice script — one line for every photo",
        key="generated_voice_script",
        height=max(180, min(420, 45 * max(len(ordered_photos), 4))),
        help=(
            "You may correct the text. Keep exactly one non-empty line for "
            "each photo and keep the same order."
        ),
    )

    script_preview_lines = [
        line.strip()
        for line in edited_script.splitlines()
        if line.strip()
    ]

    for index, line in enumerate(script_preview_lines, start=1):
        label = (
            clean_area_name(sequence_text.splitlines()[index - 1])
            if index <= len(sequence_text.splitlines())
            else "photo"
        )
        st.write(f"**Photo {index} — {label}:** {line}")

    if len(script_preview_lines) != len(ordered_photos):
        st.error(
            f"The script currently has {len(script_preview_lines)} non-empty "
            f"lines, but there are {len(ordered_photos)} photos."
        )

if script_is_current:
    st.subheader("Step 5 — Choose the voiceover method")

    voiceover_mode = st.radio(
        "How should the generated script be spoken?",
        options=[
            "Use automatic AI voice",
            "Upload my recorded voiceover",
        ],
        horizontal=True,
    )

    uploaded_voiceover_clip = None

    if voiceover_mode == "Upload my recorded voiceover":
        st.info(
            "Read the generated script from Photo 1 to the final photo in one "
            "continuous recording, then upload that recording below."
        )

        uploaded_voiceover_clip = st.file_uploader(
            "Upload your complete voiceover recording",
            type=["mp3", "wav", "m4a", "aac", "ogg"],
            accept_multiple_files=False,
            key="uploaded_final_voiceover_audio",
        )

        if uploaded_voiceover_clip is not None:
            st.audio(uploaded_voiceover_clip)

    if st.button(
        "🎬 Generate Final Video",
        type="primary",
        use_container_width=True,
    ):
        try:
            voiceovers = prepare_custom_voiceovers(
                st.session_state.get("generated_voice_script", ""),
                len(ordered_photos),
            )
            images = load_images(ordered_photos)

            if voiceover_mode == "Use automatic AI voice":
                with st.spinner(
                    "Creating AI voice and matching it with the exact photo sequence..."
                ):
                    (
                        video_bytes,
                        speed_factors,
                        line_durations,
                    ) = create_video_bytes(
                        images,
                        voiceovers,
                        PHOTO_DURATION,
                    )

                st.success(
                    "Video created with AI voice, generated script, and exact photo order."
                )

                with st.expander("AI voice and photo timing details"):
                    for index, (speed, duration) in enumerate(
                        zip(speed_factors, line_durations),
                        start=1,
                    ):
                        st.write(
                            f"Photo {index}: {duration:.2f} seconds "
                            f"at {speed:.2f}x voice speed"
                        )

            else:
                if uploaded_voiceover_clip is None:
                    st.error("Upload your recorded voiceover before generating.")
                    st.stop()

                with st.spinner(
                    "Adding your recording and matching it with the generated script..."
                ):
                    (
                        video_bytes,
                        total_audio_duration,
                        photo_durations,
                    ) = create_video_with_uploaded_voice_clip(
                        images,
                        uploaded_voiceover_clip,
                        voiceovers,
                    )

                st.success(
                    "Video created with your uploaded recording, generated script, "
                    "and exact photo order."
                )

                with st.expander("Uploaded voice and photo timing details"):
                    st.write(
                        f"Total recording duration: "
                        f"{total_audio_duration:.2f} seconds"
                    )
                    for index, duration in enumerate(
                        photo_durations,
                        start=1,
                    ):
                        st.write(
                            f"Photo {index}: {duration:.2f} seconds"
                        )

            st.subheader("Video Preview")

            left_space, preview_column, right_space = st.columns([1, 1, 1])

            with preview_column:
                try:
                    st.video(video_bytes, width=360)
                except TypeError:
                    st.video(video_bytes)

            st.download_button(
                "⬇️ Download Video",
                data=video_bytes,
                file_name="jaipur_rental_final_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        except Exception as error:
            st.exception(error)
            
            
#             #Spacious 3 BHK fully furnished flat available for rent in Ganesh Nagar, Mansarovar, Jaipur. The flat features well-designed bedrooms, air conditioning, a modern modular kitchen, comfortable living spaces, and dedicated parking. Located in a prime and peaceful area with easy access to markets, schools, hospitals, and public transport. Ideal for families looking for a ready-to-move-in home.

# Contact Jaipur Rental today for more details and a property visit.#
            