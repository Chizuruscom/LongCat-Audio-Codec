#!/bin/bash

# ==============================================================================
#                 Inference Script for LongcatAudioCodec
# ==============================================================================
#
# Description:
#   This script runs the Python inference demo, passing all necessary
#   configurations and file paths as arguments.
#
# Pre-requisite:
#   This script MUST be executed from the root of the 'longcat-codec' project
#   directory for the relative paths to work correctly.
#
# Usage:
#   bash run_inference.sh
#

# Stop the script if any command fails
set -e

echo "--- Starting LongcatAudioCodec Inference ---"

# --- 1. Configuration Variables ---
# All paths are relative to the project root ('longcat-codec/').

# Number of acoustic quantizers (codebooks) to use during encoding 
N_ACOUSTIC_CODEBOOKS=3
#! mention!!! Total codebooks used in N_ACOUSTIC_CODEBOOKS+1, since we always use one semantic codebook
TOTAL_CODEBOOKS=$((N_ACOUSTIC_CODEBOOKS+1))

# Configuration files for the models
ENCODER_CONFIG="configs/LongCatAudioCodec_encoder.yaml"
DECODER_16K_CONFIG="configs/LongCatAudioCodec_decoder_16k_4codebooks.yaml"

if [ "$TOTAL_CODEBOOKS" -eq 2 ] || [ "$TOTAL_CODEBOOKS" -eq 4 ]; then
    DECODER_24K_CONFIG=$(printf "configs/LongCatAudioCodec_decoder_24k_%scodebooks.yaml" "${TOTAL_CODEBOOKS}")
else
    DECODER_24K_CONFIG="configs/LongCatAudioCodec_decoder_24k_4codebooks.yaml"
fi

# Directory to save the reconstructed audio files
OUTPUT_DIR="demo_audio_${TOTAL_CODEBOOKS}_codebooks"

# List of input audio files to process
# The backslash '\' at the end of each line allows for a readable multi-line list.
AUDIO_FILES="demos/org/angry.wav \
             demos/org/comfort.wav \
             demos/org/common.wav \
             demos/org/fear.wav \
             demos/org/happy.wav \
             demos/org/hate.wav \
             demos/org/sad.wav \
             demos/org/sorry.wav \
             demos/org/surprise.wav"

# --- 2. Execute the Python Inference Script ---
# The backslashes '\' are used to break the long command into multiple lines for readability.
# Note that $AUDIO_FILES is NOT in quotes to allow the shell to split it into multiple arguments.

echo "Running Python script with the following parameters:"
echo "  - Encoder Config: $ENCODER_CONFIG"
echo "  - 16k Decoder Config: $DECODER_16K_CONFIG"
echo "  - 24k Decoder Config: $DECODER_24K_CONFIG"
echo "  - Output Directory: $OUTPUT_DIR"
echo "  - Num Quantizers: $N_ACOUSTIC_CODEBOOKS"
echo "  - Input Files: (list below)"
echo "$AUDIO_FILES" | sed 's/^/    - /' # Pretty print file list
echo "----------------------------------------------------"

python inference.py \
    --encoder_config "$ENCODER_CONFIG" \
    --decoder16k_config "$DECODER_16K_CONFIG" \
    --decoder24k_config "$DECODER_24K_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --n_acoustic_codebooks "$N_ACOUSTIC_CODEBOOKS" \
    --audio_files $AUDIO_FILES


# --- 3. Completion Message ---
echo "--- Inference script finished successfully! ---"
echo "Please check the output files in the '$OUTPUT_DIR' directory."