import os
import torch
import glob
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

openvoice_path = os.path.join(current_dir, 'openvoice')

# On l'ajoute à la liste des chemins que Python scanne
sys.path.append(openvoice_path)

# =================CONFIGURATION=================
# Chemins basés sur ta capture d'écran
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_openvoice")
CONVERTER_CHECKPOINT = os.path.join(MODELS_DIR, "converter", "checkpoint.pth")
CONVERTER_CONFIG = os.path.join(MODELS_DIR, "converter", "config.json")
TARGET_VOICE_FILE = os.path.join(MODELS_DIR, "target_voice.wav")

OUTPUT_DIR = os.path.join(BASE_DIR, "converted")
CLEAN_DIR = os.path.join(BASE_DIR, "clean")

# Réglage GPU / CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"🔄 Chargement du modèle OpenVoice sur {DEVICE}...")
    if not os.path.exists(CONVERTER_CHECKPOINT):
        raise FileNotFoundError(f"❌ Manquant : {CONVERTER_CHECKPOINT}")

    converter = ToneColorConverter(CONVERTER_CONFIG, device=DEVICE)
    converter.load_ckpt(CONVERTER_CHECKPOINT)
    print("✅ Modèle chargé avec succès.")
    return converter


def find_latest_clean_audio():
    # Cherche le dernier dossier créé dans 'clean'
    all_subdirs = [d for d in glob.glob(os.path.join(CLEAN_DIR, "*")) if os.path.isdir(d)]
    if not all_subdirs:
        return None
    latest_subdir = max(all_subdirs, key=os.path.getmtime)

    wav_path = os.path.join(latest_subdir, "clean.wav")
    if os.path.exists(wav_path):
        return wav_path, os.path.basename(latest_subdir)  # Retourne le chemin et le nom du dossier (timestamp)
    return None, None


def main():
    # 1. Trouver l'audio source (ta voix nettoyée)
    source_wav, timestamp_folder = find_latest_clean_audio()
    if not source_wav:
        print("⚠️ Aucun fichier 'clean.wav' trouvé. Lance d'abord TreatVoice.py")
        return

    print(f"🎙️ Audio source trouvé : {source_wav}")

    # 2. Vérifier la voix cible
    if not os.path.exists(TARGET_VOICE_FILE):
        print(f"❌ Erreur : Place un fichier audio référence ici : {TARGET_VOICE_FILE}")
        return

    # 3. Initialiser le modèle
    converter = load_model()

    # 4. Extraire l'empreinte vocale (Target Speaker Embedding)
    print("🧠 Analyse de la voix cible...")
    target_se, _ = se_extractor.get_se(TARGET_VOICE_FILE, converter, target_dir=MODELS_DIR)

    # 5. Extraire l'empreinte de ta voix (Source Speaker Embedding)
    # OpenVoice a besoin de comprendre ta voix pour la supprimer
    print("🧠 Analyse de ta voix source...")
    source_se, _ = se_extractor.get_se(source_wav, converter, target_dir=os.path.dirname(source_wav))

    # 6. Conversion
    print("🚀 Conversion en cours...")

    # Création du dossier de sortie
    save_dir = os.path.join(OUTPUT_DIR, timestamp_folder)
    os.makedirs(save_dir, exist_ok=True)
    output_filename = os.path.join(save_dir, "openvoice_result.wav")

    # Le paramètre tau contrôle la vitesse/qualité (défaut souvent 0.3 ou 1.0)
    converter.convert(
        audio_src_path=source_wav,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_filename,
        message="Status: Converting..."
    )

    print(f"🎉 Terminé ! Résultat sauvegardé ici :\n👉 {output_filename}")


if __name__ == "__main__":
    main()