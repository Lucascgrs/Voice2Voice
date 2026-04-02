import os
import sys

# Doit être avant les imports openvoice pour que commons/utils/models soient trouvables
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'openvoice'))

import torch
import glob
import shutil
import numpy as np
import soundfile as sf
import librosa
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# =================CONFIGURATION=================
MODELS_DIR = os.path.join(BASE_DIR, "models_openvoice")
TARGETS_ROOT = os.path.join(MODELS_DIR, "targets")  # Nouveau dossier racine des voix

CONVERTER_CHECKPOINT = os.path.join(MODELS_DIR, "converter", "checkpoint.pth")
CONVERTER_CONFIG = os.path.join(MODELS_DIR, "converter", "config.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "converted")
CLEAN_DIR = os.path.join(BASE_DIR, "clean")

# GPU Check
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Imports OpenVoice
openvoice_path = os.path.join(BASE_DIR, 'openvoice')
sys.path.append(openvoice_path)


def load_model():
    print(f"🔄 Chargement du modèle OpenVoice sur {DEVICE}...")
    if not os.path.exists(CONVERTER_CHECKPOINT):
        raise FileNotFoundError(f"❌ Checkpoint introuvable : {CONVERTER_CHECKPOINT}")

    converter = ToneColorConverter(CONVERTER_CONFIG, device=DEVICE)
    converter.load_ckpt(CONVERTER_CHECKPOINT)
    print("✅ Modèle chargé.")
    return converter


def select_target_voice():
    """Affiche les dossiers de voix disponibles et demande à l'utilisateur de choisir."""
    if not os.path.exists(TARGETS_ROOT):
        os.makedirs(TARGETS_ROOT, exist_ok=True)
        print(f"❌ Dossier '{TARGETS_ROOT}' créé.")
        print("👉 Mets tes dossiers de voix dedans (ex: targets/Obama/audio1.wav) et relance.")
        return None

    # Liste uniquement les dossiers
    voices = [d for d in os.listdir(TARGETS_ROOT) if os.path.isdir(os.path.join(TARGETS_ROOT, d))]

    if not voices:
        print(f"⚠️ Aucun dossier de voix trouvé dans '{TARGETS_ROOT}'.")
        print("👉 Crée un dossier par voix (ex: targets/Obama/) et mets des wav dedans.")
        return None

    print("\n🎭 CHOISIS UNE VOIX CIBLE :")
    for i, voice in enumerate(voices):
        print(f"   [{i + 1}] {voice}")

    while True:
        try:
            choice = input("\n👉 Ton choix (numéro) : ")
            index = int(choice) - 1
            if 0 <= index < len(voices):
                selected_voice = voices[index]
                return os.path.join(TARGETS_ROOT, selected_voice)
            else:
                print("❌ Numéro invalide.")
        except ValueError:
            print("❌ Entre un chiffre.")


def prepare_target_audio(voice_folder_path):
    """
    Fusionne tous les fichiers .wav du dossier choisi en un seul fichier temporaire.
    Cela permet à l'IA d'avoir une 'moyenne' de la voix sur plusieurs extraits.
    """
    print(f"🧠 Assemblage des audios pour : {os.path.basename(voice_folder_path)}")

    # Trouver tous les fichiers audio (wav, mp3, flac)
    extensions = ['*.wav', '*.mp3', '*.flac']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(voice_folder_path, ext)))

    if not files:
        print("❌ Aucun fichier audio trouvé dans ce dossier !")
        return None

    audio_segments = []
    # On utilise 22050Hz ou 44100Hz. Pour l'embedding, la cohérence compte.
    target_sr = 44100

    for file in files:
        print(f"   ➕ Ajout : {os.path.basename(file)}")
        try:
            # Librosa charge et resample automatiquement, pratique pour mixer des sources différentes
            y, _ = librosa.load(file, sr=target_sr, mono=True)
            # On enlève les silences aux extrémités pour éviter les trous
            y, _ = librosa.effects.trim(y)
            audio_segments.append(y)
            # Ajout d'un petit silence de 0.2s entre les clips pour séparer les phrases
            silence = np.zeros(int(0.2 * target_sr))
            audio_segments.append(silence)
        except Exception as e:
            print(f"   ⚠️ Impossible de lire {file} : {e}")

    if not audio_segments:
        return None

    # Concaténation
    full_audio = np.concatenate(audio_segments)

    # Sauvegarde dans un fichier temporaire
    temp_path = os.path.join(MODELS_DIR, "temp_target_combined.wav")
    sf.write(temp_path, full_audio, target_sr)

    print(f"✅ Référence assemblée ({len(full_audio) / target_sr:.1f} sec)")
    return temp_path


def find_latest_clean_audio():
    if not os.path.exists(CLEAN_DIR):
        return None, None
    all_subdirs = [d for d in glob.glob(os.path.join(CLEAN_DIR, "*")) if os.path.isdir(d)]
    if not all_subdirs:
        return None, None
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    wav_path = os.path.join(latest_subdir, "clean.wav")
    if os.path.exists(wav_path):
        return wav_path, os.path.basename(latest_subdir)
    return None, None


def main():
    # 1. Trouver l'audio source (ta voix)
    source_wav, timestamp_folder = find_latest_clean_audio()
    if not source_wav:
        print("⚠️ Aucun fichier 'clean.wav' trouvé. Lance d'abord TreatVoice.py")
        return

    # 2. Choisir la voix cible
    voice_folder = select_target_voice()
    if not voice_folder:
        return

    # 3. Préparer la référence (fusion des fichiers)
    target_wav_path = prepare_target_audio(voice_folder)
    if not target_wav_path:
        return

    # 4. Conversion
    try:
        converter = load_model()
        voice_name = os.path.basename(voice_folder)

        print(f"\n🎙️ Source : {timestamp_folder}/clean.wav")
        print(f"🎭 Cible  : {voice_name}")

        # Extraction empreintes
        target_se, _ = se_extractor.get_se(target_wav_path, converter, target_dir=MODELS_DIR)
        source_se, _ = se_extractor.get_se(source_wav, converter, target_dir=os.path.dirname(source_wav))

        print("🚀 Conversion en cours...")

        # Dossier de sortie
        save_dir = os.path.join(OUTPUT_DIR, timestamp_folder)
        os.makedirs(save_dir, exist_ok=True)

        # Nom du fichier incluant le nom de la voix cible
        output_filename = os.path.join(save_dir, f"result_{voice_name}.wav")

        converter.convert(
            audio_src_path=source_wav,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_filename,
            message="Converting",
            tau=1.0
        )

        print("-" * 50)
        print(f"🎉 Succès ! Résultat :")
        print(f"👉 {output_filename}")
        print("-" * 50)

    finally:
        # Nettoyage du fichier temporaire
        if os.path.exists(target_wav_path):
            os.remove(target_wav_path)
            # On enlève aussi le fichier .npy généré par OpenVoice s'il existe
            npy_path = target_wav_path.replace(".wav", ".npy")  # ou _se.npy
            # OpenVoice a tendance à laisser des fichiers de cache, on peut les nettoyer ici si besoin
            # Mais par sécurité on laisse le strict minimum.


if __name__ == "__main__":
    main()