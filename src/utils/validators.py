"""Input validation utilities."""

import os


def validate_input_file(file_path: str) -> bool:
    """Validate that input file exists and is readable.

    Args:
        file_path: Path to input file

    Returns:
        True if valid

    Raises:
        FileNotFoundError: If file does not exist
        PermissionError: If file is not readable
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File is not readable: {file_path}")
    return True


def validate_output_path(file_path: str) -> bool:
    """Validate output file path.

    Args:
        file_path: Path to output file

    Returns:
        True if valid

    Raises:
        PermissionError: If directory is not writable
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        raise FileNotFoundError(f"Output directory does not exist: {directory}")
    if directory and not os.access(directory, os.W_OK):
        raise PermissionError(f"Output directory is not writable: {directory}")
    return True


def validate_audio_format(file_path: str) -> bool:
    """Validate audio file format.

    Args:
        file_path: Path to audio file

    Returns:
        True if valid audio format

    Raises:
        ValueError: If format is not supported
    """
    supported_formats = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    if not file_path.lower().endswith(supported_formats):
        raise ValueError(
            f"Unsupported audio format. Supported: {supported_formats}"
        )
    return True
