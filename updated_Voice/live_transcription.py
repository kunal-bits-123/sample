import asyncio
import queue
import threading
import numpy as np
from typing import Optional, Callable
import torch
import os
import json
from datetime import datetime
import wave
import pyaudio

try:
    from pymongo import MongoClient
    from config.database_config import MONGO_CONFIG
    USE_MONGO = True
except Exception as e:
    print("MongoDB not available, using fallback storage")
    USE_MONGO = False

class AudioHandler:
    def __init__(self, callback):
        self.callback = callback
        self.chunk = 1024
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 5
        self.is_recording = False
        
        self.p = pyaudio.PyAudio()
        self.frames = []

    def start_recording(self):
        """Start recording audio"""
        self.is_recording = True
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print("* Recording audio...")
        
        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
            
            # Convert to numpy array for processing
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            # Check if we have enough data
            if len(self.frames) * self.chunk >= self.rate * self.record_seconds:
                # Process the audio
                if self.callback:
                    self.callback(np.concatenate([np.frombuffer(f, dtype=np.float32) for f in self.frames]))
                self.frames = []

    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

class LiveTranscription:
    def __init__(self, 
                 callback: Optional[Callable[[str], None]] = None,
                 model_size: str = "base",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 compute_type: str = "float16" if torch.cuda.is_available() else "int8",
                 use_fallback: bool = False):
        """
        Initialize the live transcription system
        Args:
            callback: Optional function to call with transcribed text
            model_size: Size of the Whisper model to use
            device: Device to run the model on
            compute_type: Computation type for the model
            use_fallback: Whether to use fallback storage instead of MongoDB
        """
        self.callback = callback
        self.is_running = False
        
        # Initialize Whisper model
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            self.model = None
            
        self.use_fallback = use_fallback
        
        if not use_fallback and USE_MONGO:
            # MongoDB connection for storing transcriptions
            try:
                self.mongo_client = MongoClient(
                    host=MONGO_CONFIG['host'],
                    port=MONGO_CONFIG['port']
                )
                self.db = self.mongo_client[MONGO_CONFIG['database']]
                self.collection = self.db[MONGO_CONFIG['collection']]
            except Exception as e:
                print(f"Error connecting to MongoDB: {e}")
                print("Falling back to file storage")
                self.use_fallback = True
        else:
            self.use_fallback = True
            
        if self.use_fallback:
            # Create transcripts directory if it doesn't exist
            self.transcripts_dir = os.path.join("test_data", "transcripts")
            os.makedirs(self.transcripts_dir, exist_ok=True)
            
        # Initialize audio handler
        self.audio_handler = AudioHandler(self.process_audio)

    def process_audio(self, audio_data: np.ndarray):
        """Process audio data and perform transcription"""
        if self.model is None:
            print("Whisper model not initialized")
            return
            
        try:
            segments, _ = self.model.transcribe(audio_data)
            transcribed_text = " ".join([segment.text for segment in segments])
            
            if transcribed_text.strip():
                # Save transcription
                self.save_transcription(transcribed_text)
                
                # Call callback if provided
                if self.callback:
                    self.callback(transcribed_text)
                    
        except Exception as e:
            print(f"Error processing audio: {e}")

    def save_transcription(self, text: str, metadata: dict = None):
        """Save transcription to storage"""
        if not self.use_fallback:
            try:
                document = {
                    "text": text,
                    "timestamp": datetime.now(),
                    "metadata": metadata or {}
                }
                self.collection.insert_one(document)
            except Exception as e:
                print(f"Error saving to MongoDB: {e}")
                self._save_to_file(text, metadata)
        else:
            self._save_to_file(text, metadata)

    def _save_to_file(self, text: str, metadata: dict = None):
        """Save transcription to JSON file"""
        transcripts_file = os.path.join(self.transcripts_dir, "transcripts.json")
        try:
            with open(transcripts_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"transcripts": []}

        transcript = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        data["transcripts"].append(transcript)

        with open(transcripts_file, 'w') as f:
            json.dump(data, f, indent=2)

    async def start(self):
        """Start the live transcription"""
        if self.model is None:
            print("Cannot start: Whisper model not initialized")
            return
            
        self.is_running = True
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.audio_handler.start_recording)
        self.recording_thread.start()
        
        try:
            while self.is_running:
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in transcription loop: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the live transcription"""
        self.is_running = False
        self.audio_handler.stop_recording()
        if not self.use_fallback and hasattr(self, 'mongo_client'):
            self.mongo_client.close()

# Example usage
async def print_transcription(text: str):
    print(f"Transcribed: {text}")

async def main():
    transcriber = LiveTranscription(callback=print_transcription)
    try:
        await transcriber.start()
    except KeyboardInterrupt:
        transcriber.stop()

if __name__ == "__main__":
    asyncio.run(main()) 