# tts_synthesizer.py

import pyttsx3
import platform
import sys

class TTSSynthesizer:
    def __init__(self, rate: int = None, voice_id: str = None):
        """
        Initializes the pyttsx3 TTS Synthesizer.

        Args:
            rate (int, optional): Speech rate. Defaults to pyttsx3's default.
            voice_id (str, optional): Specific voice ID to use. 
                                      Defaults to pyttsx3's default.
        """
        print("TTS (pyttsx3): Initializing...")
        self.engine = None

        try:
            # On macOS, we need to use the 'nsss' driver
            if platform.system() == "Darwin":
                self.engine = pyttsx3.init('nsss')
            else:
                self.engine = pyttsx3.init()
            
            if self.engine is None:
                print("TTS (pyttsx3): CRITICAL ERROR - pyttsx3.init() returned None. No driver loaded.")
                if platform.system() == "Linux":
                    print("TTS (pyttsx3): On Linux, ensure 'espeak' and/or 'festival' are installed.")
                elif platform.system() == "Darwin":
                    print("TTS (pyttsx3): On macOS, ensure speech synthesis services are enabled in System Settings.")
                elif platform.system() == "Windows":
                    print("TTS (pyttsx3): On Windows, ensure SAPI5 is functioning correctly.")
                return

            if rate:
                self.engine.setProperty('rate', rate)
            
            if voice_id:
                try:
                    self.engine.setProperty('voice', voice_id)
                except Exception as e_voice:
                    print(f"TTS (pyttsx3): Warning - Could not set voice ID '{voice_id}': {e_voice}")
            
            # Get current settings
            try:
                driver_name = self.engine.getProperty('driverName')
                current_rate = self.engine.getProperty('rate')
                current_voice_id = self.engine.getProperty('voice')
                print(f"TTS (pyttsx3): Engine initialized with driver: {driver_name}")
                print(f"TTS (pyttsx3): Current rate: {current_rate}, Current voice ID: {current_voice_id}")
            except Exception as e:
                print(f"TTS (pyttsx3): Warning - Could not get engine properties: {e}")

        except Exception as e:
            print(f"TTS (pyttsx3): Error initializing engine: {e}")
            self.engine = None

    def speak_text(self, text: str) -> bool:
        """
        Speaks the given text directly.
        Args:
            text (str): The text to synthesize and speak.
        Returns:
            bool: True if speech was attempted successfully, False otherwise.
        """
        if not self.engine:
            print("TTS (pyttsx3): Engine not initialized. Cannot speak.")
            return False
        
        print(f"TTS (pyttsx3): Speaking: '{text}'")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            print("TTS (pyttsx3): Finished speaking.")
            return True
        except Exception as e:
            print(f"TTS (pyttsx3): Error during speech: {e}")
            return False

    def save_to_file(self, text: str, output_path: str = "response_audio_pyttsx3.wav") -> str:
        """
        Synthesizes speech from text and saves it to an audio file.
        Args:
            text (str): The text to synthesize.
            output_path (str, optional): Path to save the output WAV file. 
        Returns:
            str: Path to the saved audio file, or an error message.
        """
        if not self.engine:
            return "TTS (pyttsx3): Engine not initialized. Cannot save to file."

        print(f"TTS (pyttsx3): Synthesizing to file for: '{text}'")
        try:
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            print(f"TTS (pyttsx3): Speech saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"TTS (pyttsx3): Error saving to file: {e}")
            return f"TTS (pyttsx3): Error - {e}"

'''

import subprocess
import platform
import logging
from typing import Optional

class TTSSynthesizer:
    def __init__(self, rate: int = 150, voice_id: Optional[str] = None):
        """Initialize the TTS synthesizer."""
        self.rate = rate
        self.voice = voice_id or "Samantha"  # Default to Samantha voice on macOS

        # Verify we're on macOS
        if platform.system() != 'Darwin':
            raise RuntimeError("This TTS implementation is for macOS only")

        self.logger = logging.getLogger("TTSSynthesizer")
        self.logger.info(f"TTS initialized with rate: {rate}")
        self.logger.info(f"Using voice: {self.voice}")

    def speak_text(self, text: str) -> bool:
        """Speak the given text using macOS 'say' command."""
        try:
            if not text:
                self.logger.warning("Empty text provided to speak_text")
                return False

            self.logger.info(f"Speaking: {text}")
            subprocess.run(['say', '-v', self.voice, '-r', str(self.rate), text])
            return True
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {e}")
            return False

    def stop(self):
        """Stop any ongoing speech."""
        try:
            subprocess.run(['pkill', 'say'])
        except Exception as e:
            self.logger.error(f"Error stopping TTS: {e}")

    def save_to_file(self, text: str, output_path: str = "response_audio.wav") -> str:
        """Save speech to a file."""
        try:
            self.logger.info(f"TTS: Synthesizing to file for: '{text}'")
            subprocess.run(['say', '-v', self.voice, '-r', str(self.rate), '-o', output_path, text])
            self.logger.info(f"TTS: Speech saved to {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving speech to file: {e}")
            return ""
'''