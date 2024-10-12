import random
import struct
from typing import Any, Union

import openai

from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.errors import ToolParameterValidationError, ToolProviderCredentialValidationError
from core.tools.tool.builtin_tool import BuiltinTool


class PodcastAudioGeneratorTool(BuiltinTool):
    def generate_silence(self, duration):
        # Generate silent MP3 data
        # This is a simplified version and may not work perfectly with all MP3 players
        # For production use, consider using a proper audio library or pre-generated silence MP3
        sample_rate = 44100
        num_samples = int(duration * sample_rate)
        silence_data = struct.pack("<" + "h" * num_samples, *([0] * num_samples))

        # Add a simple MP3 header (this is not a complete MP3 file, but might work for basic needs)
        mp3_header = b"\xff\xfb\x90\x04"  # A very basic MP3 header
        return mp3_header + silence_data

    def _invoke(
        self, user_id: str, tool_parameters: dict[str, Any]
    ) -> Union[ToolInvokeMessage, list[ToolInvokeMessage]]:
        # Extract parameters
        script = tool_parameters.get("script", "")
        host1_voice = tool_parameters.get("host1_voice")
        host2_voice = tool_parameters.get("host2_voice")

        # Split the script into lines
        script_lines = script.split("\n")

        # Ensure voices are provided
        if not host1_voice or not host2_voice:
            raise ToolParameterValidationError("Host voices are required")

        # Get OpenAI API key from credentials
        if not self.runtime or not self.runtime.credentials:
            raise ToolProviderCredentialValidationError("Tool runtime or credentials are missing")
        api_key = self.runtime.credentials.get("api_key")
        if not api_key:
            raise ToolProviderCredentialValidationError("OpenAI API key is missing")

        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)

        audio_segments = []
        for i, line in enumerate(script_lines):
            if line.strip():  # Skip empty lines
                voice = host1_voice if i % 2 == 0 else host2_voice
                try:
                    response = client.audio.speech.create(model="tts-1", voice=voice, input=line.strip())
                    audio_segments.append(response.content)

                    # Add silence between lines (except for the last line)
                    if i < len(script_lines) - 1:
                        silence_duration = random.uniform(2, 5)  # Random duration between 1 and 3 seconds
                        silence = self.generate_silence(silence_duration)
                        audio_segments.append(silence)
                except Exception as e:
                    return self.create_text_message(f"Error generating audio: {str(e)}")

        # Combine audio segments
        combined_audio = b"".join(audio_segments)

        # Create a blob message with the combined audio
        return [
            self.create_text_message("Audio generated successfully"),
            self.create_blob_message(
                blob=combined_audio,
                meta={"mime_type": "audio/mpeg"},
                save_as=self.VariableKey.AUDIO,
            ),
        ]
