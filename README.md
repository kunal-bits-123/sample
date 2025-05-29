# Clinical Voice Assistant with MCP Agents

A comprehensive clinical voice assistant system that uses multiple Medical Cognitive Processing (MCP) agents to handle various healthcare tasks through voice commands.

## System Architecture

### Core Components

1. **Voice Input Processing**
   - Live audio transcription using Whisper (large-v2 model)
   - Voice Activity Detection (VAD) for accurate speech detection
   - Real-time audio processing with PyAudio

2. **MCP Agents**
   - **EHR Agent**: Handles patient medical records and history
   - **Medication Agent**: Manages prescriptions and drug interactions
   - **Order Agent**: Processes test and procedure orders
   - **Clinical Decision Agent**: Provides clinical guidelines and protocols
   - **Scheduling Agent**: Manages appointments and availability
   - **Analytics Agent**: Generates reports and analyzes trends
   - **Inspector Agent**: Validates responses and monitors agent states

3. **Database Layer**
   - Primary: PostgreSQL for EHR data
   - Secondary: MongoDB for transcription storage
   - Fallback: File-based JSON storage

### Workflow

1. **Voice Input**
   ```
   User speaks → Audio capture → VAD processing → Whisper transcription
   ```

2. **Command Processing**
   ```
   Transcribed text → Command classification → Agent selection → Response generation
   ```

3. **Response Validation**
   ```
   Agent response → Inspector validation → State monitoring → Final output
   ```

## Setup Instructions

1. **Environment Setup**
   ```bash
   # Create and activate virtual environment
   python -m venv venv_new
   .\venv_new\Scripts\activate  # Windows
   source venv_new/bin/activate  # Linux/Mac
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_api_key
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=clinical_voice
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=your_password
   ```

3. **Database Setup**
   - PostgreSQL: Run the schema in `updated_Voice/mcp_agents/ehr_database.sql`
   - MongoDB: No additional setup required (optional)

## Usage

1. **Starting the Assistant**
   ```bash
   python clinical_voice_assistant.py
   ```

2. **Example Commands**
   - Patient Records:
     ```
     "Show me John Smith's medical history"
     "What are the patient's current medications?"
     ```
   - Medications:
     ```
     "Check interactions between Metformin and Lisinopril"
     "Prescribe antihistamine for seasonal allergies"
     ```
   - Orders:
     ```
     "Order a complete blood count test"
     "Schedule an MRI for next week"
     ```
   - Guidelines:
     ```
     "Show me the latest clinical guidelines for diabetes"
     "What's the protocol for hypertension management?"
     ```
   - Scheduling:
     ```
     "Schedule an appointment for next week"
     "Check available slots for Dr. Smith"
     ```
   - Analytics:
     ```
     "Generate a report on patient outcomes"
     "Show me the trend in blood pressure readings"
     ```

3. **Output Format**
   - Success: Formatted response with relevant information
   - Error: Clear error message with suggested actions
   - Validation: Inspector agent feedback on response quality

## Testing

1. **Unit Tests**
   ```bash
   pytest test_transcription.py
   ```

2. **Test Data**
   - Sample transcripts in `test_data/transcripts/`
   - Mock patient data in `test_data/ehr/`
   - Test audio files in `test_data/audio/`

## System Requirements

- Python 3.13+
- PostgreSQL 12+ (optional)
- MongoDB 4.4+ (optional)
- Visual C++ Build Tools (for webrtcvad)
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for faster transcription)

## Error Handling

1. **Database Failures**
   - Automatic fallback to file-based storage
   - Error logging and notification

2. **Transcription Issues**
   - VAD-based speech detection
   - Automatic retry on failure
   - Fallback to smaller model if needed

3. **Agent Errors**
   - Response validation
   - State monitoring
   - Error recovery mechanisms

## Output Examples

1. **Patient History Query**
   ```
   Input: "Show me John Smith's medical history"
   Output: 
   💬 Assistant: Patient John Smith (DOB: 1980-05-15)
   - Active Conditions:
     * Type 2 Diabetes (diagnosed 2018-03-10)
     * Hypertension (diagnosed 2019-06-22)
   - Current Medications:
     * Metformin 500mg (twice daily)
     * Lisinopril 10mg (once daily)
   - Allergies:
     * Penicillin (Moderate reaction)
   ```

2. **Medication Interaction Check**
   ```
   Input: "Check interactions between Metformin and Lisinopril"
   Output:
   💬 Assistant: No significant interactions found between Metformin and Lisinopril.
   Both medications can be safely used together. Monitor for:
   - Blood pressure changes
   - Blood glucose levels
   - Kidney function
   ```

3. **Appointment Scheduling**
   ```
   Input: "Schedule an appointment for next week"
   Output:
   💬 Assistant: Available slots for next week:
   - Monday, 10:00 AM
   - Tuesday, 2:30 PM
   - Wednesday, 11:15 AM
   Please specify your preferred time or say "book [time]"
   ```

## Troubleshooting

1. **Audio Issues**
   - Check microphone permissions
   - Verify audio device selection
   - Ensure proper audio drivers

2. **Database Connection**
   - Verify PostgreSQL/MongoDB service
   - Check connection credentials
   - Confirm network access

3. **API Issues**
   - Validate GROQ API key
   - Check API rate limits
   - Verify internet connection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details 