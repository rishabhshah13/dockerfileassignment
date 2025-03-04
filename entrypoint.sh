#!/bin/bash

# Default entrypoint mode if none specified
DEFAULT_MODE="no_quantization"

# Check if ENTRYPOINT_MODE environment variable is set
if [ -z "$ENTRYPOINT_MODE" ]; then
    # If running interactively, prompt the user
    if [ -t 0 ]; then
        echo "No ENTRYPOINT_MODE specified."
        echo "Please select an entrypoint:"
        echo "1) No Quantization (entrypoint_no_quantization.sh)"
        echo "2) Quantization (entrypoint_quantization.sh)"
        read -p "Enter 1 or 2: " choice
        case $choice in
            1)
                ENTRYPOINT_MODE="no_quantization"
                ;;
            2)
                ENTRYPOINT_MODE="quantization"
                ;;
            *)
                echo "Invalid choice. Defaulting to $DEFAULT_MODE."
                ENTRYPOINT_MODE="$DEFAULT_MODE"
                ;;
        esac
    else
        # Non-interactive mode, use default
        echo "No ENTRYPOINT_MODE specified and not running interactively. Defaulting to $DEFAULT_MODE."
        ENTRYPOINT_MODE="$DEFAULT_MODE"
    fi
fi

# Execute the selected entrypoint script
case "$ENTRYPOINT_MODE" in
    "no_quantization")
        echo "Running entrypoint_no_quantization.sh..."
        exec /entrypoint_no_quantization.sh
        ;;
    "quantization")
        echo "Running entrypoint_quantization.sh..."
        exec /entrypoint_quantization.sh
        ;;
    *)
        echo "Invalid ENTRYPOINT_MODE: $ENTRYPOINT_MODE. Valid options are 'no_quantization' or 'quantization'."
        echo "Defaulting to $DEFAULT_MODE."
        exec /entrypoint_no_quantization.sh
        ;;
esac