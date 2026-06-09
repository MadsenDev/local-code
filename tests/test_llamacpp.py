from local_code.llamacpp import format_llama_server_command, generate_llama_server_command


def test_rtx3060_command_is_generated_not_executed():
    report = generate_llama_server_command("qwen36-35b-a3b", "rtx3060", "/models/qwen.gguf")
    command = report["command"]
    assert "llama-server" in command
    assert "-m /models/qwen.gguf" in command
    assert "-c 16384" in command
    assert "-ngl 999" in command
    assert "--n-cpu-moe 30" in command
    assert "--host 127.0.0.1" in command
    assert "--port 8080" in command
    assert "does not execute" in format_llama_server_command(report)
