import os
import argparse
from typing import List, Tuple, Generator

import torch
import torchaudio

from networks.semantic_codec.model_loader import load_encoder, load_decoder


def wav_list_generator(
    file_list: List[str], batch_size: int, sample_rate: int, device: torch.device
) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[str]], None, None]:
    """
    A generator that loads, pre-processes, and batches audio files.

    This function takes a list of audio file paths and groups them into batches.
    For each batch, it performs the following steps:
    1. Loads the audio waveform.
    2. Resamples to the target `sample_rate` if necessary.
    3. Converts to mono by averaging channels.
    4. Pads all waveforms in the batch to the length of the longest one.

    Args:
        file_list (List[str]): A list of paths to the audio files.
        batch_size (int): The number of audio files in each batch.
        sample_rate (int): The target sample rate to resample all audio to.
        device (torch.device): The device (e.g., "cuda" or "cpu") to place tensors on.

    Yields:
        Tuple[torch.Tensor, torch.Tensor, List[str]]: A tuple containing:
        - A padded batch of audio waveforms with shape (Batch, 1, Length).
        - A tensor of the original, un-padded waveform lengths.
        - A list of the file paths included in the current batch.
    """
    for i in range(0, len(file_list), batch_size):
        batch_keys = file_list[i:i + batch_size]
        wavs, wav_lens = [], []
        max_len = 0
        
        for path in batch_keys:
            wav, sr = torchaudio.load(path)
            # Resample if the sample rate does not match the target
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sample_rate)
            # Convert to mono by averaging channels if necessary
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            wavs.append(wav)
            wav_lens.append(wav.shape[-1])
            if wav.shape[-1] > max_len:
                max_len = wav.shape[-1]
        
        # Initialize a zero tensor to hold the padded batch
        padded_wavs = torch.zeros(len(wavs), 1, max_len)
        # Copy each waveform into the padded tensor
        for j, wav in enumerate(wavs):
            padded_wavs[j, 0, :wav.shape[-1]] = wav
            
        yield padded_wavs.to(device), torch.LongTensor(wav_lens).to(device), batch_keys


def main():
    """
    Main function to run the codec demonstration.
    This script showcases two primary uses of the codec:
    1. Reconstructing audio from tokens at different sample rates.
    2. Using the API to extract semantic and acoustic tokens from audio files.
    """
    # --- Argument Parsing and Configuration ---
    parser = argparse.ArgumentParser(
        description="Demo script for a semantic-acoustic neural audio codec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    parser.add_argument(
        '--encoder_config', 
        type=str, 
        required=True, 
        help="Path to the encoder's YAML configuration file."
    )
    parser.add_argument(
        '--decoder16k_config', 
        type=str, 
        required=True, 
        help="Path to the 16kHz decoder's YAML configuration file."
    )
    parser.add_argument(
        '--decoder24k_config', 
        type=str, 
        required=True, 
        help="Path to the 24kHz decoder's YAML configuration file."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True, 
        help="Directory where the output audio will be saved."
    )
    parser.add_argument(
        '--audio_files', 
        nargs='+',  
        required=True, 
        help="One or more paths to the input audio files."
    )
    parser.add_argument(
        '--n_acoustic_codebooks', 
        type=int, 
        default=2, 
        help="Number of acoustic codebooks (quantizers) to use for encoding. More quantizers generally means higher quality."
    )

    args = parser.parse_args()

    # --- Setup Environment and Models ---
    encoder_config_path = args.encoder_config
    decoder16k_config_path = args.decoder16k_config
    decoder24k_config_path = args.decoder24k_config
    output_dir = args.output_dir
    audio_files = args.audio_files
    n_acoustic_codebooks = args.n_acoustic_codebooks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the pre-trained encoder and multi-rate decoders
    encoder = load_encoder(encoder_config_path, device)
    decoder16k = load_decoder(decoder16k_config_path, device)
    decoder24k = load_decoder(decoder24k_config_path, device)

    # --- Demo 1: Reconstruct audio at different sample rates from the same tokens ---
    # This demo shows how to encode an audio file into a set of tokens and then
    # use different decoders to synthesize high-quality audio at various sample rates.
    print("\n" + "="*50)
    print("Demo 1: Multi-rate Audio Synthesis from a Single Set of Tokens")
    print("="*50)
    with torch.no_grad():
        for audio_path in audio_files:
            print(f"\nProcessing file: {audio_path}")
            base_filename = os.path.basename(audio_path)
            output_path_16k = os.path.join(output_dir, base_filename.replace('.wav', '_reconstructed_16k.wav'))
            output_path_24k = os.path.join(output_dir, base_filename.replace('.wav', '_reconstructed_24k.wav'))
            
            wav, sr = torchaudio.load(audio_path)
            # Add a batch dimension and move the tensor to the target device
            wav = wav.unsqueeze(0).to(device)

            # Encode the audio waveform. This returns a tuple containing:
            # 0: A tensor of semantic tokens (capturing content).
            # 1: A tensor of acoustic tokens (capturing sound quality).
            # The number of acoustic codebooks (`n_acoustic_codebooks`) can be adjusted
            # to control the quality/bitrate trade-off.
            codes = encoder(wav, sr, n_acoustic_codebooks=n_acoustic_codebooks)

            # Decode the tokens back into audio at 16kHz and 24kHz respectively.
            reconstructed_wav_16k = decoder16k(codes[0], codes[1])
            reconstructed_wav_24k = decoder24k(codes[0], codes[1])
            
            # Save the reconstructed audio files
            torchaudio.save(output_path_16k, reconstructed_wav_16k.cpu().squeeze(0), decoder16k.output_rate)
            print(f"  - Saved 16kHz reconstructed audio to: {output_path_16k}")
            
            torchaudio.save(output_path_24k, reconstructed_wav_24k.cpu().squeeze(0), decoder24k.output_rate)
            print(f"  - Saved 24kHz reconstructed audio to: {output_path_24k}")
            
    # --- Demo 2: API for extracting semantic and acoustic tokens ---
    # This demo shows how to use the encoder's API to extract semantic and
    # acoustic tokens separately for a batch of audio files.
    print("\n" + "="*50)
    print("Demo 2: Batch Token Extraction API")
    print("="*50)
    # Create a data generator to process audio files in batches.
    # It handles loading, resampling to the encoder's expected input rate, and padding.
    batch_generator = wav_list_generator(audio_files, batch_size=4, sample_rate=encoder.input_sample_rate, device=device)
    
    with torch.no_grad():
        for wavs_batch, wav_lens_batch, keys_batch in batch_generator:
            print(f"\n--- Processing a new batch of {len(keys_batch)} files ---")
            print(f"  - Padded audio tensor shape: {wavs_batch.shape}")
            print(f"  - Original lengths tensor: {wav_lens_batch.tolist()}")
            
            # Pre-process the raw audio batch to match the encoder's expected format.
            # `input_wav_sample_rate` should match the sample rate of the loaded audio.
            input_wav_sample_rate = 16000 
            wavs_batch = encoder.preprocess(wavs_batch, input_wav_sample_rate)

            # API Call 1: Extract acoustic tokens.
            # The second return value, `codes_lens`, indicates the number of valid
            # tokens for each audio file in the batch before padding.
            acoustic_codes, codes_lens = encoder.get_acoustic_codes_with_lengths(wavs_batch, wav_lens_batch, n_acoustic_codebooks=n_acoustic_codebooks)
            
            # API Call 2: Extract semantic tokens.
            # The token lengths are the same as for the acoustic codes, so we can discard them here.
            semantic_codes, _ = encoder.get_semantic_codes_with_lengths(wavs_batch, wav_lens_batch)
            
            print(f"  - Valid Token Lengths (codes_lens): {codes_lens.tolist()}")
            print(f"  - Extracted Semantic Codes Shape: {semantic_codes.shape}")
            if acoustic_codes is not None:
                 print(f"  - Extracted Acoustic Codes Shape: {acoustic_codes.shape}")

if __name__ == "__main__":
    main()