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


def test_dense_coder_profile_skips_moe_flags():
    report = generate_llama_server_command("qwen2.5-coder-7b", "vram12", "/models/qwen7.gguf")
    command = report["command"]
    assert "--n-cpu-moe" not in command
    assert "Qwen2.5 Coder 7B" in format_llama_server_command(report)


def test_capability_gpu_profile_generates_command():
    report = generate_llama_server_command("qwen2.5-coder-14b", "vram24", "/models/qwen14.gguf")
    command = report["command"]
    assert "-c 32768" in command
    assert "-b 2048" in command
