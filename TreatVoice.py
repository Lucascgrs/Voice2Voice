import os
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, lfilter

# ============================================================
#                     CONFIGURATION GLOBALE
# ============================================================

TARGET_SR = 44100
MAX_PEAK = 0.98


# ============================================================
#                FILTRE PASSE-HAUT
# ============================================================

def high_pass_filter(audio, sr, cutoff=30):
    """Supprime les infrabasses (<30Hz)."""
    b, a = butter(2, cutoff / (sr / 2), btype='highpass')
    return lfilter(b, a, audio)


# ============================================================
#                REDUCTION DU BRUIT (DOUCE)
# ============================================================

def denoise(audio, sr):
    """
    Réduction de bruit optimisée pour la voix naturelle.
    stationary=True est plus sûr pour éviter les artefacts métalliques.
    prop_decrease=0.6 enlève 60% du bruit (compromis idéal).
    """
    try:
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=0.6,
            stationary=True,
            n_fft=2048
        )
        return reduced_noise
    except Exception as e:
        print(f"⚠️ Warning noisereduce: {e}")
        return audio


# ============================================================
#                  SUPPRESSION DES SILENCES
# ============================================================

def remove_silence(audio, sr, threshold=45):
    """
    Découpe les silences.
    threshold=45dB : Conserve les respirations et fins de phrases douces.
    frame_length/hop_length : Assure des coupures fluides.
    """
    # Détection des parties non silencieuses
    clips = librosa.effects.split(
        audio,
        top_db=threshold,
        frame_length=2048,
        hop_length=512
    )

    if len(clips) == 0:
        return audio

    # Reconstruction
    processed = np.concatenate([audio[start:end] for start, end in clips])
    return processed


# ============================================================
#                    NORMALISATION
# ============================================================

def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio * (MAX_PEAK / peak)


# ============================================================
#                 PIPELINE DE TRAITEMENT
# ============================================================

def preprocess_file(input_path, output_path):
    print(f"🔧 Traitement : {os.path.basename(input_path)}")

    try:
        # Charger l’audio
        audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

        # 1. Filtre passe-haut
        audio = high_pass_filter(audio, sr)

        # 2. Réduction de bruit (Douce)
        audio = denoise(audio, sr)

        # 3. Suppression des silences (Intelligente)
        audio = remove_silence(audio, sr)

        # 4. Normalisation
        audio = normalize(audio)

        # Sauvegarde
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, audio, TARGET_SR, subtype="PCM_16")

        print(f"   ✅ Succès -> {output_path}")

    except Exception as e:
        print(f"   ❌ Erreur sur {input_path} : {e}")


# ============================================================
#                     MAIN
# ============================================================

def preprocess_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ Dossier '{folder_path}' introuvable.")
        return

    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                input_file = os.path.join(root, file)
                # Structure miroir dans le dossier clean
                relative = os.path.relpath(root, folder_path)
                output_dir = os.path.join("clean", relative)
                output_file = os.path.join(output_dir, "clean.wav")

                preprocess_file(input_file, output_file)
                count += 1

    if count == 0:
        print("⚠️ Aucun fichier .wav trouvé à traiter.")


if __name__ == "__main__":
    preprocess_folder("recordings")