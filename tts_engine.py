from gtts import gTTS

def text_to_audio(text, output_path):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_path)
    return output_path