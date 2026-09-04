"""Main dubbing pipeline orchestration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.config.settings import Config
from src.core.audio_processor import AudioProcessor
from src.core.job_state import JobController, PipelineRequest
from src.core.orchestrator import DubbingOrchestrator
from src.models.voice_synthesis import VoiceSynthesis
from src.utils.validators import validate_input_file


class DubbingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.voice_synthesis = VoiceSynthesis(config)
        self.orchestrator = DubbingOrchestrator(config)

    def process(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        source_language: str = "auto",
        target_language: str = "en",
        callback=None,
        controller: Optional[JobController] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        validate_input_file(input_file)
        result = self.orchestrator.run(
            PipelineRequest(
                input_file=input_file,
                source_language=source_language,
                target_language=target_language,
                output_dir=output_file,
            ),
            callback=callback,
            controller=controller,
        )
        if output_file:
            result["output"] = output_file
        return result
