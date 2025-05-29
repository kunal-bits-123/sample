import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

class FallbackHandler:
    def __init__(self):
        """Initialize the fallback handler using JSON files"""
        self.test_data_dir = "test_data"
        self.ehr_dir = os.path.join(self.test_data_dir, "ehr")
        self.transcripts_dir = os.path.join(self.test_data_dir, "transcripts")
        
        # Create directories if they don't exist
        os.makedirs(self.ehr_dir, exist_ok=True)
        os.makedirs(self.transcripts_dir, exist_ok=True)
        
        # Initialize test data if it doesn't exist
        self._init_test_data()

    def _init_test_data(self):
        """Initialize test data if it doesn't exist"""
        # Check if patients.json exists
        patients_file = os.path.join(self.ehr_dir, "patients.json")
        if not os.path.exists(patients_file):
            with open(patients_file, 'w') as f:
                json.dump({"patients": []}, f)

        # Create transcripts file if it doesn't exist
        transcripts_file = os.path.join(self.transcripts_dir, "transcripts.json")
        if not os.path.exists(transcripts_file):
            with open(transcripts_file, 'w') as f:
                json.dump({"transcripts": []}, f)

    def save_transcription(self, text: str, metadata: dict = None):
        """Save transcription to JSON file"""
        transcripts_file = os.path.join(self.transcripts_dir, "transcripts.json")
        try:
            with open(transcripts_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"transcripts": []}

        transcript = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        data["transcripts"].append(transcript)

        with open(transcripts_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_patient_records(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get patient records from JSON file"""
        patients_file = os.path.join(self.ehr_dir, "patients.json")
        try:
            with open(patients_file, 'r') as f:
                data = json.load(f)
                for patient in data.get("patients", []):
                    if patient["id"] == patient_id:
                        return [
                            {
                                "record_type": "condition",
                                "data": condition
                            } for condition in patient.get("medical_history", {}).get("conditions", [])
                        ] + [
                            {
                                "record_type": "medication",
                                "data": medication
                            } for medication in patient.get("current_medications", [])
                        ]
        except FileNotFoundError:
            return []
        return []

    def add_patient(self, patient_data: Dict[str, Any]) -> str:
        """Add a new patient to JSON file"""
        patients_file = os.path.join(self.ehr_dir, "patients.json")
        try:
            with open(patients_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"patients": []}

        # Generate a new patient ID
        patient_id = f"P{len(data['patients']) + 1:03d}"
        patient_data["id"] = patient_id
        data["patients"].append(patient_data)

        with open(patients_file, 'w') as f:
            json.dump(data, f, indent=2)

        return patient_id

    def close(self):
        """No need to close anything for file-based storage"""
        pass 