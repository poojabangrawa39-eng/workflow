import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

# Read API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please add it to your .env file."
    )

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


def generate_script(prompt, num_photos):
    """
    Generate narration segments for each property photo.
    Returns a list of narration strings.
    """

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert real estate marketing copywriter. "
                    "Write engaging, professional property narration."
                )
            },
            {
                "role": "user",
                "content": f"""
Create a voiceover script for a real estate property video.

Property Description:
{prompt}

Number of Photos:
{num_photos}

Instructions:
- Return exactly {num_photos} narration segments.
- Number each segment.
- Each segment should contain 1–2 short sentences.
- Keep the tone professional, engaging, and natural.
- Highlight important property features.
- Do not add introductions or conclusions.
"""
            }
        ]
    )

    raw = response.choices[0].message.content.strip()

    segments = []

    for line in raw.split("\n"):
        line = line.strip()

        if not line:
            continue

        if line[0].isdigit():
            segment = line.split(".", 1)[-1].strip()
            segments.append(segment)

    # Ensure the number of segments matches the number of photos
    if len(segments) < num_photos:
        while len(segments) < num_photos:
            segments.append("Beautiful property with excellent features.")

    return segments[:num_photos]