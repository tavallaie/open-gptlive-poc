"""Gradio entry point for a Hugging Face Space."""

import gradio as gr

try:
    from client import LiveDemoClient, convert_to_pcm16le
except ImportError:
    from .client import LiveDemoClient, convert_to_pcm16le


def run_demo(audio):
    """Convert the recording, call the optional endpoint, and render results."""
    if audio is None:
        return "Record or upload audio first.", "", None
    try:
        result = LiveDemoClient().run(convert_to_pcm16le(audio))
    except ValueError as exc:
        return str(exc), "", None
    return result.status, result.transcript, result.audio


with gr.Blocks(title="GPT-Live demo") as demo:
    gr.Markdown("# GPT-Live audio demo\nRecord audio and optionally send it to a configured GPT-Live WebSocket endpoint.")
    audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Input audio")
    run_button = gr.Button("Send")
    status = gr.Textbox(label="Status")
    transcript = gr.Textbox(label="Transcript")
    audio_output = gr.Audio(label="Output audio", type="numpy")
    run_button.click(run_demo, inputs=audio_input, outputs=[status, transcript, audio_output])


if __name__ == "__main__":
    demo.launch()
