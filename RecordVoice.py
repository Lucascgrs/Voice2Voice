import sounddevice as sd
import numpy as np
import soundfile as sf
import datetime
import os
import sys

# ============================================================
#                CONFIGURATION GLOBALE
# ============================================================

SAMPLE_RATE = 44100  # Qualité CD
CHANNELS = 1  # Mono (meilleur pour l'IA)
SUBTYPE = "PCM_16"  # WAV 16 bits

is_recording = False  # Indicateur global
audio_buffer = []  # Stockage temporaire
max_peak = 0.0  # Pour suivre le volume max atteint


# ============================================================
#           FONCTION : VISUALISATION (VU-MÈTRE)
# ============================================================

def print_vu_meter(indata):
    """Affiche une barre de volume dans la console."""
    global max_peak
    volume_norm = np.linalg.norm(indata) * 10
    peak = np.max(np.abs(indata))

    if peak > max_peak:
        max_peak = peak

    # Création de la barre visuelle
    bars = int(volume_norm)
    bar_str = '█' * bars
    pad_str = ' ' * (50 - bars)

    # Indicateur de saturation (clipping)
    warn = " ⚠️ SATURATION !" if peak >= 0.99 else ""

    # \r permet de réécrire sur la même ligne
    sys.stdout.write(f"\rVolume: |{bar_str}{pad_str}| {peak:.2f} {warn}")
    sys.stdout.flush()


# ============================================================
#           CALLBACK AUDIO
# ============================================================

def audio_callback(indata, frames, time, status):
    if status:
        print(f"\n⚠️ Erreur Audio: {status}")
    if is_recording:
        audio_buffer.append(indata.copy())
        print_vu_meter(indata)


# ============================================================
#           LANCEMENT DE L'ENREGISTREMENT
# ============================================================

def start_recording():
    global is_recording, audio_buffer, max_peak

    audio_buffer = []
    max_peak = 0.0
    is_recording = True

    print("=" * 60)
    print("🎙️  ENREGISTREMENT LANCÉ")
    print("Instructions :")
    print("1. Parle normalement.")
    print("2. Essaie de garder la barre de volume vers le milieu.")
    print("3. Appuie sur [ENTRÉE] pour arrêter.")
    print("=" * 60)

    # Ouvre le flux audio
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        input()  # Attend que l'utilisateur appuie sur Entrée

    is_recording = False
    print("\n\n🛑 Enregistrement terminé.")

    # Bilan de qualité
    print(f"📊 Volume Max atteint : {max_peak:.2f}")
    if max_peak > 0.98:
        print("❌ ATTENTION : L'audio a saturé. Parle moins fort la prochaine fois.")
    elif max_peak < 0.1:
        print("⚠️ ATTENTION : Volume très faible. Rapproche-toi du micro.")
    else:
        print("✅ Niveau de volume excellent.")

    save_recording(audio_buffer)


# ============================================================
#           SAUVEGARDE
# ============================================================

def save_recording(buffer):
    if not buffer:
        print("❌ Aucun audio enregistré.")
        return

    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join("recordings", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, "audio.wav")

    audio_array = np.concatenate(buffer, axis=0)
    sf.write(file_path, audio_array, SAMPLE_RATE, subtype=SUBTYPE)

    print(f"💾 Fichier sauvegardé : {file_path}")


if __name__ == "__main__":
    try:
        start_recording()
    except KeyboardInterrupt:
        print("\n\nArrêt forcé.")