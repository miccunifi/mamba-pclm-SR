"""
YouTube -> MP3
Download the audio of a YouTube video and convert it to MP3.

Requirements:
    pip install yt-dlp
    FFmpeg installed on the system (needed for the MP3 conversion)
      - Windows: download it from https://ffmpeg.org and add it to the PATH
      - macOS:   brew install ffmpeg
      - Linux:   sudo apt install ffmpeg

Usage:
    python youtube_to_mp3.py
    (the script asks for the video link)

    or directly from the command line:
    python youtube_to_mp3.py "https://www.youtube.com/watch?v=XXXXXXX"
"""

import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    sys.exit("Errore: yt-dlp non è installato. Esegui: pip install yt-dlp")


def scarica_mp3(url: str, cartella_output: str = "downloads") -> None:
    """Download the audio of the video and convert it to MP3."""
    Path(cartella_output).mkdir(parents=True, exist_ok=True)

    opzioni = {
        # Pick the best available audio stream
        "format": "bestaudio/best",
        # File name: the video title
        "outtmpl": str(Path(cartella_output) / "%(title)s.%(ext)s"),
        # Convert to MP3 with FFmpeg
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",  # kbps: 128, 192, 256 or 320 are valid
            }
        ],
        "noplaylist": True,   # download only the video, not the whole playlist
        "quiet": False,       # show progress
    }

    # Download and convert; report failures as messages instead of tracebacks
    try:
        with yt_dlp.YoutubeDL(opzioni) as ydl:
            info = ydl.extract_info(url, download=True)
            titolo = info.get("title", "video")
            print(f"\n✅ Fatto! '{titolo}.mp3' salvato nella cartella '{cartella_output}'")
    except yt_dlp.utils.DownloadError as e:
        print(f"\n❌ Errore durante il download: {e}")
    except Exception as e:
        print(f"\n❌ Errore imprevisto: {e}")


def main() -> None:
    # Use the link from the command line, or ask the user for it
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Inserisci il link del video YouTube: ").strip()

    if not url:
        sys.exit("Nessun link inserito. Uscita.")

    scarica_mp3(url)


if __name__ == "__main__":
    main()