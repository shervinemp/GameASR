#!/usr/bin/env python3
"""
Setup script for Text-to-Speech (TTS) system using Kokoro-ONNX.

This script handles installing dependencies and setting up the environment.
"""

import os


def download_file(url, destination):
    """
    Download a file from a URL to a specified destination.
    """
    import requests

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {url} to {destination}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


def setup_environment():
    """
    Set up the environment for TTS by creating necessary directories.
    """
    # Create the models directory if it doesn't exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        try:
            os.makedirs(models_dir)
            print(f"Created directory: {models_dir}")
        except Exception as e:
            print(f"Failed to create directory {models_dir}: {e}")


def main():
    """
    Main function to set up the TTS environment.
    """
    # Set up the environment (create directories)
    setup_environment()

    # Check if required files exist
    required_files = [
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            # Download the file if it does not exist
            destination = os.path.join("models", os.path.basename(file_path))
            download_file(file_path, destination)
        else:
            print(f"File {file_path} already exists, skipping download.")

    print("TTS setup completed successfully.")


if __name__ == "__main__":
    main()
