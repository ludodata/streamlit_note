import streamlit as st
from pytube import YouTube
import os

# Définir l'emplacement de destination par défaut sur le bureau
st.set_option('deprecation.showfileUploaderEncoding', False)  # Désactiver l'encodage

# Titre de l'application
st.title("Télécharger l'audio d'une vidéo Youtube")
st.subheader("Auteur : Peter Hook")

# Ajouter une zone de texte pour l'URL YouTube
url = st.text_input("Entrez l'URL de la vidéo YouTube:")

# Ajouter un bouton de téléchargement de l'audio
if st.button("Télécharger l'audio"):
    if url:
        try:
            # Télécharger la vidéo depuis YouTube
            st.write("Téléchargement de la video Youtube en cours...")
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()  # Sélectionner le premier flux audio disponible

            # Spécifier le dossier de destination sur le bureau de l'utilisateur
            destination_folder = os.path.join(os.path.expanduser("~"), "Desktop", "zik")

            # Créer le dossier de destination s'il n'existe pas
            os.makedirs(destination_folder, exist_ok=True)


             # Vérifier si le fichier existe déjà
            audio_path = os.path.join(destination_folder, f"{yt.title}.mp3")
            if os.path.exists(audio_path):
                raise FileExistsError("Le fichier existe déjà. Veuillez supprimer l'ancien fichier ou le déplacer.")

            # Télécharger l'audio
            st.write("Conversion de l'audio en cours...")
            audio_path = os.path.join(destination_folder, f"{yt.title}.mp3")
            audio_stream.download(output_path=destination_folder)

            # Renommer le fichier au format MP3
            os.rename(os.path.join(destination_folder, audio_stream.default_filename), audio_path)

            # Afficher le lien de téléchargement de l'audio
            st.success("Conversion de l'audio réussi !")
            st.write("Retrouve la chanson dans le dossier zik sur ton bureau !")
            

        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")
    else:
        st.warning("Veuillez entrer une URL YouTube avant de télécharger l'audio.")
