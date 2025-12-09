import sounddevice as sd
import numpy as np
import soundfile as sf
import datetime
import os

# ============================================================
#                CONFIGURATION GLOBALE
# ============================================================

SAMPLE_RATE = 44100        # Qualité CD
CHANNELS = 1               # Mono (meilleur pour training IA)
SUBTYPE = "PCM_16"         # WAV 16 bits

is_recording = False       # Indicateur global
audio_buffer = []          # Stockage temporaire des frames


# ============================================================
#           FONCTION : ENREGISTREMENT AUDIO CONTINU
# ============================================================

def audio_callback(indata, frames, time, status):
    """Callback appelé à chaque chunk audio reçu."""
    if status:
        print("⚠️ Status:", status)
    if is_recording:
        audio_buffer.append(indata.copy())


# ============================================================
#           FONCTION : LANCER L’ENREGISTREMENT
# ============================================================

def start_recording():
    global is_recording, audio_buffer

    audio_buffer = []  # reset
    is_recording = True

    print("🎙️ Enregistrement lancé… Parle ! (appuie sur Entrée pour arrêter)")

    # Ouvre un flux audio
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback
    )

    stream.start()

    # Attendre que l’utilisateur appuie sur Entrée
    input()

    # Quand l'utilisateur veut arrêter :
    is_recording = False
    stream.stop()
    stream.close()

    print("🛑 Enregistrement arrêté.")

    # Sauvegarde
    save_recording(audio_buffer)


# ============================================================
#           FONCTION : SAUVEGARDER LE FICHIER WAV
# ============================================================

def save_recording(buffer):
    # Création du dossier horodaté
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join("recordings", folder_name)

    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, "audio.wav")

    # =======================================================
    # Reconstruction du tableau audio depuis les frames
    # =======================================================
    if len(buffer) == 0:
        print("⚠️ Aucun audio dans le buffer !")
        return

    # Concaténer correctement les blocs audio
    audio_array = np.concatenate(buffer, axis=0)

    # =======================================================
    # Sauvegarde du WAV proprement
    # =======================================================
    sf.write(file_path, audio_array, 44100, subtype="PCM_16")

    print(f"💾 Audio sauvegardé : {file_path}")


# ============================================================
#                     POINT D’ENTRÉE
# ============================================================

if __name__ == "__main__":
    start_recording()
