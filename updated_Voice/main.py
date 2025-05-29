import os
import time
import json
import asyncio
import torch
import config

from STT import STTProcessor
from nlp_handler import NLPHandler
from TTS import TTSSynthesizer


conversation_log_detailed = []  # Structured conversation log


def setup_pytorch_device_and_dtype():
    resolved_device_str = "cpu"
    if config.DEVICE_PREFERENCE.lower() == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            resolved_device_str = "mps"
        elif torch.cuda.is_available():
            resolved_device_str = "cuda:0"
    elif config.DEVICE_PREFERENCE.lower().startswith("cuda"):
        if torch.cuda.is_available():
            resolved_device_str = config.DEVICE_PREFERENCE.lower()
            try:
                if not torch.cuda.is_available_for_device(torch.device(resolved_device_str)):
                    resolved_device_str = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
            except Exception:
                resolved_device_str = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            resolved_device_str = "mps"

    resolved_dtype = torch.float32
    if (resolved_device_str.startswith("cuda") or resolved_device_str == "mps") and \
            config.TORCH_DTYPE_PREFERENCE.lower() == "float16":
        resolved_dtype = torch.float16

    config.RESOLVED_PYTORCH_DEVICE = resolved_device_str
    config.RESOLVED_PYTORCH_DTYPE = resolved_dtype
    print(f"MAIN: PyTorch Device: {resolved_device_str}, Dtype: {resolved_dtype}")


def load_encounter_log():
    global conversation_log_detailed
    conversation_log_detailed = []
    if not os.path.exists(config.CONVERSATION_HISTORY_FILE):
        print(f"MAIN: No encounter log found at {config.CONVERSATION_HISTORY_FILE}. Starting fresh.")
        return

    try:
        with open(config.CONVERSATION_HISTORY_FILE, "r", encoding="utf-8") as f:
            current_speaker = None
            current_text_block = []
            for line in f:
                line_strip = line.strip()
                if line_strip.startswith("Doctor-Patient Transcript:") or line_strip.startswith("Assistant Analysis:"):
                    if current_speaker and current_text_block:
                        conversation_log_detailed.append({
                            "speaker": current_speaker,
                            "text": "\n".join(current_text_block).strip()
                        })
                    current_speaker = line_strip.split(":")[0]
                    current_text_block = [line_strip.replace(f"{current_speaker}: ", "", 1).strip()]
                elif current_speaker and line_strip:
                    current_text_block.append(line_strip)
                elif current_speaker and not line_strip and current_text_block:
                    conversation_log_detailed.append({
                        "speaker": current_speaker,
                        "text": "\n".join(current_text_block).strip()
                    })
                    current_speaker = None
                    current_text_block = []

            if current_speaker and current_text_block:
                conversation_log_detailed.append({
                    "speaker": current_speaker,
                    "text": "\n".join(current_text_block).strip()
                })

        print(f"MAIN: Loaded encounter log with {len(conversation_log_detailed)} entries.")
    except Exception as e:
        print(f"MAIN: Error loading encounter log: {e}")


def save_to_encounter_log(entry_type: str, content):
    global conversation_log_detailed
    if not content:
        return

    if isinstance(content, dict):
        text_to_log = json.dumps(content, indent=2)
    else:
        text_to_log = str(content)

    conversation_log_detailed.append({"speaker": entry_type, "text": text_to_log})

    try:
        with open(config.CONVERSATION_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"{entry_type}:\n{text_to_log}\n\n")
        print(f"MAIN: Saved to encounter log: {entry_type}")
    except Exception as e:
        print(f"MAIN: Error saving to encounter log: {e}")


async def process_conversation_segment(transcript: str, nlp_module: NLPHandler, tts_module: TTSSynthesizer):
    if not transcript:
        print("MAIN: No transcript to process.")
        tts_module.speak_text("I didn't catch any conversation. Please try recording again.")
        return

    print(f"MAIN: Doctor-Patient Transcript:\n{transcript}")
    save_to_encounter_log("Doctor-Patient Transcript", transcript)

    print("MAIN: Analyzing conversation (NLP)...")
    start_time = time.time()
    analysis_result = await nlp_module.analyze_conversation_async(transcript)
    print(f"MAIN: NLP analysis took {time.time() - start_time:.2f}s")

    summary = "Could not analyze the conversation."
    if analysis_result and not analysis_result.get("error"):
        save_to_encounter_log("Assistant Analysis", analysis_result)

        parts = []
        if analysis_result.get("chief_complaint"):
            parts.append(f"The chief complaint appears to be {analysis_result['chief_complaint']}.")
        if analysis_result.get("symptoms"):
            parts.append(f"Symptoms mentioned include {', '.join(analysis_result['symptoms'])}.")
        if analysis_result.get("proposed_plan_orders"):
            parts.append(f"The proposed plan includes {', '.join(analysis_result['proposed_plan_orders'])}.")

        if parts:
            summary = " ".join(parts) + " Is there anything else I can note or do?"
        else:
            summary = "I've processed the conversation. What would you like to do next?"
    elif analysis_result and analysis_result.get("error"):
        error_msg = analysis_result.get("details", "Unknown NLP error")
        print(f"MAIN: NLP error: {error_msg}")
        summary = f"Sorry, I had trouble analyzing the conversation. NLP error: {analysis_result.get('error')}"
        save_to_encounter_log("Assistant Analysis Error", summary)

    print(f"MAIN: Assistant summary to speak: '{summary}'")

    print("MAIN: Speaking assistant's summary (TTS)...")
    start_time = time.time()
    success = tts_module.speak_text(summary)
    print(f"MAIN: TTS speaking took {time.time() - start_time:.2f}s")
    if not success:
        print("MAIN: TTS failed to speak summary.")
    else:
        print("MAIN: Assistant's summary spoken.")


async def main_async_flow():
    print("Starting Clinical Conversation Assistant...")
    start_time = time.time()

    setup_pytorch_device_and_dtype()
    load_encounter_log()

    stt_module = None
    nlp_module = None
    tts_module = None

    try:
        print("Initializing STT (Faster Whisper & VAD)...")
        stt_module = STTProcessor(
            model_size=config.FASTER_WHISPER_MODEL_SIZE,
            device=config.FASTER_WHISPER_DEVICE,
            compute_type=config.FASTER_WHISPER_COMPUTE_TYPE,
            sample_rate=config.VAD_SAMPLE_RATE,
            block_size=config.VAD_BLOCK_SIZE,
            energy_threshold=config.VAD_ENERGY_THRESHOLD,
            silence_timeout_s=config.VAD_SILENCE_TIMEOUT_S,
            language=config.STT_LANGUAGE,
            output_audio_filename=config.CONVERSATION_RECORDING_FILENAME
        )
        if not stt_module.whisper_model:
            print("MAIN: STT model failed to initialize. Exiting.")
            return

        print("Initializing NLP (Groq)...")
        if config.NLP_API_PROVIDER != "groq":
            print(f"MAIN: Unexpected NLP provider '{config.NLP_API_PROVIDER}', expected 'groq'. Exiting.")
            return
        if not config.GROQ_API_KEY:
            print("MAIN: GROQ_API_KEY not set. Exiting.")
            return

        nlp_module = NLPHandler(
            provider=config.NLP_API_PROVIDER,
            api_key=config.GROQ_API_KEY,
            groq_model_identifier=config.GROQ_MODEL_IDENTIFIER
        )

        print("Initializing TTS (pyttsx3)...")
        tts_module = TTSSynthesizer(
            rate=config.PYTTSX3_RATE,
            voice_id=config.PYTTSX3_VOICE_ID
        )
        if not tts_module.engine:
            print("MAIN: TTS engine failed to initialize. Exiting.")
            return

        while True:
            action = input("\nPress Enter to START recording or type 'quit' to exit: ").strip().lower()
            if action == "quit":
                print("MAIN: Exiting by user command.")
                break

            print("MAIN: Recording conversation now...")
            transcript = stt_module.record_and_transcribe_conversation_segment()
            if transcript is None:
                print("MAIN: Recording or transcription failed.")
                tts_module.speak_text("Sorry, there was an issue with recording the conversation.")
            elif transcript.strip() == "":
                print("MAIN: No speech detected.")
                tts_module.speak_text("I didn't record any conversation. Please try again.")
            else:
                await process_conversation_segment(transcript, nlp_module, tts_module)

            print("-" * 50)

    except KeyboardInterrupt:
        print("\nMAIN: KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        print(f"MAIN: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if nlp_module and hasattr(nlp_module, "close_clients"):
            print("MAIN: Closing NLP clients...")
            await nlp_module.close_clients()

        print(f"Voice Assistant finished in {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    try:
        asyncio.run(main_async_flow())
    except KeyboardInterrupt:
        print("\nMAIN: Interrupted by user (Ctrl+C).")
    finally:
        print("MAIN: Application exiting.")
