import asyncio
import json
import os
from datetime import datetime

class MockTranscription:
    def __init__(self, callback=None):
        self.callback = callback
        self.is_running = False
        
        # Create test data directory
        self.test_data_dir = "test_data/transcripts"
        os.makedirs(self.test_data_dir, exist_ok=True)
        
    async def save_transcription(self, text: str):
        """Save transcription to JSON file"""
        transcripts_file = os.path.join(self.test_data_dir, "transcripts.json")
        try:
            with open(transcripts_file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"transcripts": []}

        transcript = {
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
        data["transcripts"].append(transcript)

        with open(transcripts_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        if self.callback:
            await self.callback(text)

    async def start(self):
        """Simulate transcription with test data"""
        self.is_running = True
        test_phrases = [
            "Patient shows symptoms of seasonal allergies",
            "Blood pressure is 120 over 80",
            "Recommend follow-up in two weeks",
            "Prescribing antihistamine medication"
        ]
        
        print("Starting mock transcription...")
        try:
            for phrase in test_phrases:
                if not self.is_running:
                    break
                await self.save_transcription(phrase)
                print(f"Transcribed: {phrase}")
                await asyncio.sleep(2)  # Simulate processing time
        except Exception as e:
            print(f"Error in transcription: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the transcription"""
        self.is_running = False
        print("Stopped transcription")

async def print_transcription(text: str):
    print(f"Callback received: {text}")

async def main():
    # Test the transcription
    transcriber = MockTranscription(callback=print_transcription)
    try:
        await transcriber.start()
    except KeyboardInterrupt:
        transcriber.stop()

if __name__ == "__main__":
    asyncio.run(main()) 