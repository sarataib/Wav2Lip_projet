import os
import subprocess
import uuid
import shutil
import numpy as np
from scipy.io import wavfile
import gradio as gr

# Répertoires
WAV2LIP_DIR = "/content/drive/MyDrive/Wav2Lip"
CHECKPOINT_PATH = os.path.join(WAV2LIP_DIR, "checkpoints/wav2lip_gan.pth")
RESULTS_DIR = os.path.join(WAV2LIP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def lip_sync_with_upload(face_media_path, audio_file):
    uid = str(uuid.uuid4())
    input_video_path = os.path.join(WAV2LIP_DIR, f"temp_video_{uid}.mp4")
    input_audio_path = os.path.join(WAV2LIP_DIR, f"temp_audio_{uid}.wav")
    output_video_path = os.path.join(RESULTS_DIR, f"result_voice_{uid}.mp4")

    if face_media_path.endswith((".jpg", ".jpeg", ".png")):
        temp_image_path = os.path.join(WAV2LIP_DIR, f"temp_image_{uid}.png")
        shutil.copy(face_media_path, temp_image_path)
        cmd = f"ffmpeg -y -loop 1 -i {temp_image_path} -t 5 -vf scale=512:512 -c:v libx264 -pix_fmt yuv420p {input_video_path}"
        subprocess.call(cmd, shell=True)
        os.remove(temp_image_path)
    else:
        shutil.copy(face_media_path, input_video_path)

    if isinstance(audio_file, tuple):
        sample_rate, audio_data = audio_file
    else:
        import librosa
        audio_data, sample_rate = librosa.load(audio_file, sr=None)
        audio_data = (audio_data * 32767).astype(np.int16)

    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data, dtype=np.int16)

    wavfile.write(input_audio_path, sample_rate, audio_data)

    command = f"""
    cd {WAV2LIP_DIR} && python inference.py \\
    --checkpoint_path "{CHECKPOINT_PATH}" \\
    --audio "{input_audio_path}" \\
    --face "{input_video_path}" \\
    --outfile "{output_video_path}"
    """
    subprocess.call(command, shell=True)
    os.remove(input_video_path)
    os.remove(input_audio_path)

    return output_video_path

# Interface Gradio
with gr.Blocks(title="Wav2Lip Sync App") as interface:
    gr.HTML("""
    <style>
        body {
            background-color: #f0f2f5;
            font-family: 'Segoe UI', sans-serif;
        }
        .gr-button {
            background: linear-gradient(to right, #1d976c, #93f9b9);
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
        }
        .gr-button:hover {
            background: linear-gradient(to right, #11998e, #38ef7d);
        }
        h2 {
            color: #2e8b57;
        }
    </style>
    """)

    gr.Markdown("## 🎬 **Synchronisation Labiale Professionnelle avec Wav2Lip**")

    with gr.Row():
        with gr.Column(scale=1):
            media_input = gr.File(label="📹 Image ou Vidéo", file_types=[".jpg", ".png", ".mp4"], type="filepath")
            audio_input = gr.Audio(label="🎵 Audio (.wav)", type="filepath")

            image_preview = gr.Image(label="🖼️ Aperçu de l'image", visible=False, height=180)
            video_preview = gr.Video(label="🎞️ Aperçu de la vidéo", visible=False, height=300, width=300)

            def preview_media(file_path):
                if file_path.endswith((".jpg", ".png", ".jpeg")):
                    return gr.update(visible=True, value=file_path), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True, value=file_path)

            media_input.change(fn=preview_media, inputs=media_input, outputs=[image_preview, video_preview])

            btn = gr.Button("🚀 Lancer la Synchronisation", elem_classes="gr-button")

        with gr.Column(scale=1):
            output_video = gr.Video(label="📤 Vidéo Générée", height=360, width=700)

        btn.click(fn=lip_sync_with_upload, inputs=[media_input, audio_input], outputs=output_video)

interface.launch(share=True)
