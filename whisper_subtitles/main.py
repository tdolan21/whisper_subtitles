from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from moviepy.editor import VideoFileClip
from whisper_subtitles.utils import check_gpu_availability
from pyannote.audio import Pipeline
from rich.console import Console
from rich.layout import Layout
from dotenv import load_dotenv
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import numpy as np
import argparse
import librosa
import torch
import time 
import os
import sys
import io

# Assuming AVAILABLE_MODELS is defined as before
AVAILABLE_MODELS = [
    "openai/whisper-large-v3",
    "openai/whisper-large-v2",
    "distil-whisper/distil-large-v2",
    "distil-whisper/distil-large",
    "distil-whisper/distil-medium.en",
    "distil-whisper/distil-small.en"
]

console = Console()

def create_epilog_panel():
    epilog_text = Text("\nExample of use:\n", style="bold green")
    epilog_text.append("whisper-subtitles files/filename\n\n", style="bold cyan")
    epilog_text.append("whisper-subtitles files/filename --max_new_tokens 512 --chunk_length_s 15 --batch_size 32 --language en --translate\n\n", style="bold cyan")
    epilog_text.append("If you would like to use a different whisper model you can do so by passing it as an argument:\n", style="bold green")
    epilog_text.append("whisper-subtitles --model_id distil-whisper/distil-large-v2\n\n", style="bold cyan")
    epilog_text.append("You can specify an individual audio file or a directory containing multiple audio files. The tool supports additional options like setting the language and enabling translation.\n", style="bold green")

    return Panel(epilog_text, title="[bold magenta]Whisper Subtitles CLI[/bold magenta]", subtitle="[italic magenta]Command Examples and Usage[/italic magenta]", border_style="bright_blue")

def create_models_panel():
    # Whisper Models Table
    models_table = Table(title="Available Whisper Models", title_style="bold yellow", border_style="bright_green")
    models_table.add_column("Model ID", justify="left", style="cyan", no_wrap=True)
    for model in AVAILABLE_MODELS:
        models_table.add_row(model)

    # Create and return a Panel containing the models table
    return Panel(models_table, title="[bold magenta]Whisper Models[/bold magenta]", border_style="bright_blue")

def display_info_panels():
    epilog_panel = create_epilog_panel()
    models_panel = create_models_panel()

    layout = Layout()
    layout.split_row(
        Layout(epilog_panel, size=60),
        Layout(models_panel, size=40)
    )
    console.print(layout)
    
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Whisper Subtitles CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('path', nargs='?', type=str, default=None, help='Path to an audio file or folder containing audio files')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum new tokens for ASR model (default: 512)')
    parser.add_argument('--chunk_length_s', type=int, default=15, help='Chunk length in seconds for ASR model (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (default: 32)')
    parser.add_argument('--language', type=str, default=None, help='Language for transcription (default: None)')
    parser.add_argument('--translate', action='store_true', help='Enable translation. If specified, the transcription will be translated (default: False)')
    parser.add_argument('--model_id', type=str, default="openai/whisper-large-v3", help='Whisper model ID (default: openai/whisper-large-v3)')
    parser.add_argument('--info', action='store_true', help='Display information about models and GPU')
    parser.add_argument('--gpu', action='store_true', help='Check if GPU is available')
    
    return parser.parse_args()


def load_whisper_model(model_id, torch_dtype, max_new_tokens, chunk_length_s, batch_size):
    device = torch.device("cuda")
    torch_dtype = torch.float16

    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="flash_attention_2",
    ).to(device)
    
    console.print(f"[bold cyan] Whisper device after moving: [/bold cyan][bold green]{whisper_model.device}[/bold green]")
    #whisper_model.to_bettertransformer()
    whisper_processor = AutoProcessor.from_pretrained(model_id)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        max_new_tokens=max_new_tokens,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    return whisper_pipe


def diarization_pipeline(pipeline_id):
    auth_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    device = torch.device("cuda")
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            pipeline_id, use_auth_token=auth_token
        ).to(device)
        console.print(f"[bold cyan]Diarization Model device after moving: [/bold cyan][bold green]{diarization_pipeline.device}[/bold green]")
        return diarization_pipeline
    except Exception as e:
        console.print(f"[bold red]Failed to load diarization pipeline: {e}[/bold red]")
        return None


def resample_audio(audio, sr, target_sr=16000):
    """
    Resample the given audio to the target sampling rate.

    Parameters:
    audio (numpy.ndarray): The input audio data.
    sr (int): The current sampling rate of the audio.
    target_sr (int): The desired target sampling rate.

    Returns:
    tuple: The resampled audio and the new sampling rate.
    """
    try:
        # Check if the audio is mono or stereo
        if len(audio.shape) > 1:
            # If stereo, average the channels to convert to mono
            audio = np.mean(audio, axis=1)

        # Normalize the audio to -1 to 1 range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Resample the audio if the sampling rates differ
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        return audio, target_sr

    except Exception as e:
        # Handle exceptions and errors
        print(f"Error during resampling: {e}")
        return None, None
    
def transcribe_and_diarize(audio_file_path, max_new_tokens, chunk_length_s, batch_size, language=None, translate=False, model_id="openai/whisper-large-v3"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Load and resample audio for Whisper
        audio, sr = librosa.load(audio_file_path, sr=None)
        resampled_audio, resampled_sr = resample_audio(audio, sr)
        if resampled_audio is None or resampled_sr is None:
            raise ValueError("Failed to resample audio.")

        waveform_tensor_float32 = torch.tensor(resampled_audio, dtype=torch.float32).to(device)
        
        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        if translate:
            generate_kwargs["task"] = "translate"
        
        # Load the diarization pipeline from cache and run on audio file
        diarization = diarization_pipeline("pyannote/speaker-diarization-3.1")
        if diarization is None:
            raise ValueError("Failed to load diarization pipeline.")
        
        with ProgressHook() as hook:
            diarization.to(device)
            diarized_result = diarization(audio_file_path, hook=hook)
            if not diarized_result:
                raise ValueError("Failed to perform diarization.")

        # Load the Whisper pipeline with specified parameters
        whisper_pipe = load_whisper_model(model_id, torch.float16, max_new_tokens, chunk_length_s, batch_size)
        if whisper_pipe is None:
            raise ValueError("Failed to load Whisper pipeline.")

        # Constants for chunking
        CHUNK_DURATION, OVERLAP_DURATION = 30, 5  # in seconds
        combined_transcriptions = []

        # Process each turn in the diarization result
        for turn, _, speaker in diarized_result.itertracks(yield_label=True):
            speaker_transcription = ""
            speaker_timestamps = []

            for chunk_start in range(int(turn.start * resampled_sr), int(turn.end * resampled_sr), int(CHUNK_DURATION * resampled_sr)):
                chunk_end = min(chunk_start + int(CHUNK_DURATION * resampled_sr), int(turn.end * resampled_sr))

                speaker_audio_segment = waveform_tensor_float32[chunk_start:chunk_end].to(torch.float16).to(device)
                speaker_audio_segment_np = speaker_audio_segment.cpu().numpy()

                # Process the audio segment with Whisper pipeline
                result = whisper_pipe(
                    speaker_audio_segment_np,
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs
                )

                if not result or 'text' not in result:
                    continue

                # Aggregate transcribed text and timestamps for each chunk
                speaker_transcription += result['text'] + " "
                speaker_timestamps.extend(result['chunks'])

            # Append the aggregated transcription and timestamps for the entire turn
            combined_transcriptions.append((speaker, speaker_transcription.strip(), speaker_timestamps))

        return combined_transcriptions

    except Exception as e:
        print(f"Error in processing: {e}")
        return []
    
def create_subtitle_content(transcriptions):
    subtitle_content = io.StringIO()
    counter = 1
    for speaker, _, timestamps in transcriptions:
        for timestamp in timestamps:
            start_time = format_timestamp(timestamp['timestamp'][0])
            end_time = format_timestamp(timestamp['timestamp'][1])
            subtitle_text = f"{speaker}: {timestamp['text']}"
            subtitle_content.write(f"{counter}\n")
            subtitle_content.write(f"{start_time} --> {end_time}\n")
            subtitle_content.write(f"{subtitle_text}\n\n")
            counter += 1
    return subtitle_content.getvalue()

def format_timestamp(seconds):
    if seconds is None:
        return "00:00:00,000"
    
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

def save_subtitle_file(audio_file_path, subtitle_content):
    subtitle_file_path = audio_file_path.rsplit('.', 1)[0] + '.srt'
    with open(subtitle_file_path, 'w') as subtitle_file:
        subtitle_file.write(subtitle_content)
    return f"Subtitles saved to: {subtitle_file_path}", subtitle_file_path


def is_audio_file(filename):
    return filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))

def is_movie_file(filename):
    return filename.lower().endswith(('.mp4', '.mov', '.avi'))

# Parent function for handling audio files
def process_file(file_path, max_new_tokens, chunk_length_s, batch_size, language, translate, model_id):
    try:
        # Pass model_id to transcribe_and_diarize
        transcriptions = transcribe_and_diarize(
            file_path, 
            max_new_tokens=max_new_tokens, 
            chunk_length_s=chunk_length_s, 
            batch_size=batch_size, 
            language=language, 
            translate=translate,
            model_id=model_id  # New addition
        )
        if transcriptions:
            subtitle_content = create_subtitle_content(transcriptions)
            save_subtitle_file(file_path, subtitle_content)
            return f"Transcriptions processed for file {file_path}"
        else:
            return f"No transcriptions were found for file {file_path}."
    except Exception as e:
        return f"Error processing file {file_path}: {e}"

def process_movie_file(movie_file_path, max_new_tokens, chunk_length_s, batch_size, language, translate, model_id):
    try:
        # Ensure the movie file path is valid
        if not os.path.exists(movie_file_path):
            raise FileNotFoundError(f"Movie file not found: {movie_file_path}")

        # Define the output audio file path
        audio_file_path = movie_file_path.rsplit('.', 1)[0] + '.mp3'

        # Extract audio from the video file
        with VideoFileClip(movie_file_path) as video:
            audio = video.audio
            
            # Writing the audio file with specific parameters
            audio.write_audiofile(audio_file_path, fps=16000, codec='libmp3lame', bitrate='192k')

        # Assuming process_file is another function that processes the extracted audio file
    
        return process_file(audio_file_path, max_new_tokens, chunk_length_s, batch_size, language, translate, model_id)        
    except Exception as e:
        return f"Error processing movie file {movie_file_path}: {e}"



def display_first_15_lines_of_srt(srt_file_path):
    try:
        with open(srt_file_path, 'r') as file:
            lines = [next(file) for _ in range(15)]
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading subtitle file: {e}"


def print_in_panel(text, style=""):
    panel = Panel(text, style=style)
    console.print(panel)

def main():
    args = parse_arguments()

    if args.info:
        display_info_panels()
        return
    
    if args.gpu:
        check_gpu_availability()
        return
    
    input_path = args.path
    max_new_tokens = args.max_new_tokens
    chunk_length_s = args.chunk_length_s
    batch_size = args.batch_size
    language = args.language
    translate = args.translate
    model_id = args.model_id 

    start_time = time.time()  # Start the timer

    output_messages = []
    srt_file_path = None
    
    if os.path.isfile(input_path):
        if is_audio_file(input_path):
            output_messages.append(f"Processing audio file: {input_path}")
            message = process_file(input_path, max_new_tokens, chunk_length_s, batch_size, language, translate, model_id)
            output_messages.append(message)
        elif is_movie_file(input_path):
            output_messages.append(f"Processing movie file: {input_path}")
            message = process_movie_file(input_path, max_new_tokens, chunk_length_s, batch_size, language, translate, model_id)
            output_messages.append(message)
        else:
            output_messages.append("Unsupported file format.")
    elif os.path.isdir(input_path):
        output_messages.append(f"Processing all files in directory: {input_path}")
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if is_audio_file(filename):
                output_messages.append(f"Processing audio file: {file_path}")
                message = process_file(file_path, max_new_tokens, chunk_length_s, batch_size, language, translate)
                output_messages.append(message)
            elif is_movie_file(filename):
                output_messages.append(f"Processing movie file: {file_path}")
                message = process_movie_file(file_path, max_new_tokens, chunk_length_s, batch_size, language, translate)
                output_messages.append(message)
    else:
        output_messages.append("The provided path is neither a file nor a directory.")

    # Combine all messages and print them in a panel
    combined_output = "\n".join(output_messages)
    print_in_panel(combined_output, style="bold cyan")

    end_time = time.time()  
    duration = end_time - start_time  
    print_in_panel(f"Transcription completed in {duration:.2f} seconds.", style="bold purple")  # Print the duration
    
    if srt_file_path:
        srt_preview = display_first_15_lines_of_srt(srt_file_path)
        print_in_panel(srt_preview, style="bold cyan")

if __name__ == "__main__":
    main()