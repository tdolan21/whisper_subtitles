from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
import argparse
from colorama import Fore, Style
import os
import torch
import librosa
import numpy as np
import time 
import sys
import io

load_dotenv()

def print_colored_epilog():
    epilog = (
        f"{Fore.GREEN}Example of use:{Style.RESET_ALL}\n"
        f"  {Fore.CYAN}whisper-subtitles files/filename{Style.RESET_ALL}\n\n"
        f"  {Fore.CYAN}whisper-subtitles files/filename --max_new_tokens 512 --chunk_length_s 15 --batch_size 32 --language en --translate{Style.RESET_ALL}\n\n"
        f"{Fore.GREEN}You can specify an individual audio file or a directory containing multiple audio files. "
        f"The tool supports additional options like setting the language and enabling translation.{Style.RESET_ALL}"
    )
    print(epilog)


def parse_arguments():
    description = (
        "Whisper Subtitles\n\n"
        "This tool transcribes audio files and performs speaker diarization."
    )
   
    print_colored_epilog()
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('path', type=str, help='Path to an audio file or folder containing audio files')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum new tokens for ASR model (default: 512)')
    parser.add_argument('--chunk_length_s', type=int, default=15, help='Chunk length in seconds for ASR model (default: 15)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (default: 32)')
    parser.add_argument('--language', type=str, default=None, help='Language for transcription (default: None)')
    parser.add_argument('--translate', action='store_true', help='Enable translation. If specified, the transcription will be translated (default: False)')


    return parser.parse_args()



def check_gpu_availability():
    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        print("CUDA (GPU support) is available on this device.")
        print(f"PyTorch version: {torch.__version__}")

        # Displaying number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        # Displaying information about each GPU
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Setting default CUDA device (optional, for multi-GPU environments)
        torch.cuda.set_device(0)  # Sets the default GPU as GPU 0
        print(f"Current CUDA device index: {torch.cuda.current_device()}")
    else:
        print("CUDA (GPU support) is not available on this device.")


def load_whisper_model(model_id, torch_dtype, max_new_tokens, chunk_length_s, batch_size):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model_id = "openai/whisper-large-v3"

    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)
    print(f"Model device after moving: {whisper_model.device}")
    whisper_model.to_bettertransformer()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            pipeline_id, use_auth_token=auth_token
        ).to(device)
        print(f"Diarization Model device after moving: {diarization_pipeline.device}")
        return diarization_pipeline
    except Exception as e:
        print(f"Failed to load diarization pipeline: {e}")
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
    
def transcribe_and_diarize(audio_file_path, max_new_tokens, chunk_length_s, batch_size, language=None, translate=False):
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
        
        diarization.to(device)
        diarized_result = diarization(audio_file_path)
        if not diarized_result:
            raise ValueError("Failed to perform diarization.")

        # Load the Whisper pipeline with specified parameters
        whisper_pipe = load_whisper_model("openai/whisper-large-v3", torch.float16, max_new_tokens, chunk_length_s, batch_size)
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
    # Converts a timestamp in seconds to the SRT format (HH:MM:SS,MS)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


def process_file(file_path, max_new_tokens, chunk_length_s, batch_size, language, translate):
    try:
        transcriptions = transcribe_and_diarize(
            file_path, 
            max_new_tokens=max_new_tokens, 
            chunk_length_s=chunk_length_s, 
            batch_size=batch_size, 
            language=language, 
            translate=translate
        )

        if transcriptions:
            subtitle_content = create_subtitle_content(transcriptions)
            save_subtitle_file(file_path, subtitle_content)
        else:
            print(Fore.YELLOW + f"No transcriptions were found for file {file_path}." + Style.RESET_ALL)

    except Exception as e:
        print(Fore.RED + f"Error processing file {file_path}: {e}" + Style.RESET_ALL)

def process_movie_file(movie_file_path, max_new_tokens, chunk_length_s, batch_size, language, translate):
    try:
        # Extract audio from movie file
        with VideoFileClip(movie_file_path) as video:
            audio = video.audio
            audio_file_path = movie_file_path.rsplit('.', 1)[0] + '.mp3'
            audio.write_audiofile(audio_file_path, fps=16000, codec='libmp3lame')

        # Process the extracted audio file
        process_file(audio_file_path, max_new_tokens, chunk_length_s, batch_size, language, translate)

    except Exception as e:
        print(Fore.RED + f"Error processing movie file {movie_file_path}: {e}" + Style.RESET_ALL)

def save_subtitle_file(audio_file_path, subtitle_content):
    subtitle_file_path = audio_file_path.rsplit('.', 1)[0] + '.srt'
    with open(subtitle_file_path, 'w') as subtitle_file:
        subtitle_file.write(subtitle_content)
    print(Fore.GREEN + f"Subtitles saved to: {subtitle_file_path}" + Style.RESET_ALL)
    

def main():
    args = parse_arguments()
    input_path = args.path
    max_new_tokens = args.max_new_tokens
    chunk_length_s = args.chunk_length_s
    batch_size = args.batch_size
    language = args.language
    translate = args.translate

    start_time = time.time()  # Start the timer

    args = parse_arguments()
    # Function to determine if the file is an audio file
    def is_audio_file(filename):
        return filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))

    # Function to determine if the file is a movie file
    def is_movie_file(filename):
        return filename.lower().endswith(('.mp4', '.mov', '.avi'))

    # Check if input is a file or a directory
    if os.path.isfile(input_path):
        if is_audio_file(input_path):
            print(Fore.GREEN + "Processing audio file: " + input_path + Style.RESET_ALL)
            process_file(input_path, max_new_tokens, chunk_length_s, batch_size, language, translate)
        elif is_movie_file(input_path):
            print(Fore.GREEN + "Processing movie file: " + input_path + Style.RESET_ALL)
            process_movie_file(input_path, max_new_tokens, chunk_length_s, batch_size, language, translate)
        else:
            print(Fore.RED + "Unsupported file format." + Style.RESET_ALL)
    elif os.path.isdir(input_path):
        print(Fore.BLUE + "Processing all files in directory: " + input_path + Style.RESET_ALL)
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if is_audio_file(filename):
                print(Fore.GREEN + "Processing audio file: " + file_path + Style.RESET_ALL)
                process_file(file_path, max_new_tokens, chunk_length_s, batch_size, language, translate)
            elif is_movie_file(filename):
                print(Fore.GREEN + "Processing movie file: " + file_path + Style.RESET_ALL)
                process_movie_file(file_path, max_new_tokens, chunk_length_s, batch_size, language, translate)
    else:
        print(Fore.RED + "The provided path is neither a file nor a directory." + Style.RESET_ALL)
    end_time = time.time()  # End the timer
    duration = end_time - start_time  # Calculate the duration
    print(f"{Fore.BLUE}Transcription completed in {duration:.2f} seconds.{Style.RESET_ALL}")  # Print the duration

if __name__ == "__main__":
    main()