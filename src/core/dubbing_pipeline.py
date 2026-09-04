"""Main pipeline orchestration facade."""

from __future__ import annotations

from typing import Optional, Dict, Any

from src.config.settings import Config
from src.core.audio_processor import AudioProcessor
from src.models.voice_synthesis import VoiceSynthesis
from src.pipeline.orchestrator import DubbingOrchestrator
from src.utils.validators import validate_input_file


class DubbingPipeline:
    """Legacy + production pipeline facade."""

    def __init__(self, config: Config):
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.voice_synthesis = VoiceSynthesis(config)
        self.orchestrator = DubbingOrchestrator(config)

    def process(self, input_file: str, output_file: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        validate_input_file(input_file)

        if kwargs.get("mode") == "legacy":
            audio_data = self.audio_processor.load_audio(input_file)
            dubbed_audio = self.voice_synthesis.synthesize(audio_data, **kwargs)
            if output_file:
                self.audio_processor.save_audio(dubbed_audio, output_file)
            return {
                "status": "success",
                "input": input_file,
                "output": output_file,
                "duration": len(dubbed_audio) / self.config.sample_rate,
            }

        source_language = kwargs.get("source_language", "en")
        target_language = kwargs.get("target_language", "es")
        result = self.orchestrator.process(
            input_video=input_file,
            source_language=source_language,
            target_language=target_language,
            progress=kwargs.get("progress_callback"),
        )
        if output_file and result.get("output_video"):
            from pathlib import Path
            import shutil

            target = Path(output_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result["output_video"], target)
            result["output_video"] = str(target)
        return result
