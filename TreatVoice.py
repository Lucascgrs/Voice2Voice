import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, lfilter


# ============================================================
#                     CONFIGURATION GLOBALE
# ============================================================

TARGET_SR = 44100    # fréquence d’échantillonnage standard pour l'entraînement RVC
MAX_PEAK = 0.98      # normalisation maximale


# ============================================================
#                FILTRE PASSE-HAUT (anti-ronflement)
# ============================================================

def high_pass_filter(audio, sr, cutoff=30):
    """Supprime les très basses fréquences inutiles (<30Hz)."""
    b, a = butter(2, cutoff / (sr / 2), btype='highpass')
    return lfilter(b, a, audio)


# ============================================================
#                REDUCTION DU BRUIT (noisereduce)
# ============================================================

def denoise(audio, sr):
    """Réduction de bruit basée sur la méthode spectral gating."""
    reduced_noise = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=0.8,
        stationary=False
    )
    return reduced_noise


# ============================================================
#                  SUPPRESSION DES SILENCES
# ============================================================

def remove_silence(audio, sr, threshold=20):
    """
    Découpe les silences automatiquement.
    threshold en dB — 20dB est une valeur douce et efficace.
    """
    clips = librosa.effects.split(audio, top_db=threshold)

    processed = np.concatenate([audio[start:end] for start, end in clips])
    return processed


# ============================================================
#                    NORMALISATION AUDIO
# ============================================================

def normalize(audio):
    """Normalise le volume à MAX_PEAK (-1 à 1)."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio * (MAX_PEAK / peak)


# ============================================================
#                 PIPELINE COMPLET DE TRAITEMENT
# ============================================================

def preprocess_file(input_path, output_path):
    print(f"🔧 Prétraitement : {input_path}")

    # Charger l’audio
    audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

    # 1. Filtre passe-haut pour enlever les grondements
    audio = high_pass_filter(audio, sr)

    # 2. Réduction de bruit
    audio = denoise(audio, sr)

    # 3. Suppression des silences
    audio = remove_silence(audio, sr)

    # 4. Normalisation
    audio = normalize(audio)

    # Sauvegarde du fichier propre
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, TARGET_SR, subtype="PCM_16")

    print(f"✅ Audio nettoyé enregistré : {output_path}")


# ============================================================
#                     TRAITER UN DOSSIER ENTIER
# ============================================================

def preprocess_folder(folder_path):
    """
    Traite automatiquement tous les WAV du dossier :
    recordings/xxxx/audio.wav → clean/xxxx/clean.wav
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                input_file = os.path.join(root, file)
                relative = os.path.relpath(root, folder_path)
                output_dir = os.path.join("clean", relative)
                output_file = os.path.join(output_dir, "clean.wav")

                preprocess_file(input_file, output_file)


# ============================================================
#                     SCRIPT LANCEUR
# ============================================================

if __name__ == "__main__":
    preprocess_folder("recordings")
