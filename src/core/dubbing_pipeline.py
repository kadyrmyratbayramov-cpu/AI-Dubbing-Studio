"""Main dubbing pipeline orchestration."""

from typing import Optional, Dict, Any
from src.config.settings import Config
from src.core.audio_processor import AudioProcessor
from src.models.voice_synthesis import VoiceSynthesis
from src.utils.validators import validate_input_file


class DubbingPipeline:
    """Main pipeline for dubbing and voice synthesis workflow."""

    def __init__(self, config: Config):
        """Initialize the dubbing pipeline.

        Args:
            config: Configuration object
        """
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.voice_synthesis = VoiceSynthesis(config)

    def process(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process input file through dubbing pipeline.

        Args:
            input_file: Path to input audio/video file
            output_file: Path to output dubbing file
            **kwargs: Additional processing parameters

        Returns:
            Processing result dictionary
        """
        # Validate input
        validate_input_file(input_file)

        # Process audio
        audio_data = self.audio_processor.load_audio(input_file)

        # Synthesize voice
        dubbed_audio = self.voice_synthesis.synthesize(audio_data, **kwargs)

        # Save output if specified
        if output_file:
            self.audio_processor.save_audio(dubbed_audio, output_file)

        return {
            "status": "success",
            "input": input_file,
            "output": output_file,
            "duration": len(dubbed_audio) / self.config.sample_rate
        }
