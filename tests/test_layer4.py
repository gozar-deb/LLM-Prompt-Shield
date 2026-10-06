"""Unit tests for app.layers.layer4_output_dlp and app.core.canary."""
from app.core.canary import check_leak, generate_canary, inject_into_messages
from app.layers.layer4_output_dlp import StreamingDLPBuffer, scan_complete_output


def test_canary_roundtrip():
    canary = generate_canary()
    messages = inject_into_messages([{"role": "user", "content": "hi"}], canary)
    assert messages[0]["role"] == "system"
    assert canary.token in messages[0]["content"]
    assert check_leak(f"the secret token is {canary.token}", canary)
    assert not check_leak("nothing sensitive here", canary)


def test_scan_complete_output_blocks_on_canary_leak():
    canary = generate_canary()
    result = scan_complete_output(f"Sure, the secret token is {canary.token}", canary)
    assert result.blocked
    assert result.canary_leak
    assert result.block_reason == "canary_token_leak_detected"


def test_scan_complete_output_allows_clean_text():
    canary = generate_canary()
    result = scan_complete_output("The mitochondria is the powerhouse of the cell.", canary)
    assert not result.blocked


def test_scan_complete_output_redacts_secret_without_blocking():
    canary = generate_canary()
    result = scan_complete_output("here is a key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789", canary)
    assert not result.blocked
    assert "[REDACTED_SECRET]" in result.sanitized_text


def test_streaming_buffer_strict_mode_never_leaks_split_canary():
    """Regression test for the spec's Canary Leak Test: a token split across
    two SSE chunks must never appear in what gets released to the client."""
    canary = generate_canary()
    buf = StreamingDLPBuffer(canary, buffer_chars=16, hold_back=True)
    half = len(canary.token) // 2
    chunks = ["The answer is: " + canary.token[:half], canary.token[half:] + " — done."]

    released = []
    blocked = False
    for chunk in chunks:
        text, is_blocked, reason = buf.feed(chunk)
        released.append(text)
        if is_blocked:
            blocked = True
            break
    if not blocked:
        text, is_blocked, reason = buf.flush()
        released.append(text)
        blocked = is_blocked

    assert blocked
    assert canary.token not in "".join(released)


def test_streaming_buffer_fast_mode_still_detects_the_leak():
    canary = generate_canary()
    buf = StreamingDLPBuffer(canary, buffer_chars=16, hold_back=False)
    half = len(canary.token) // 2
    chunks = ["The answer is: " + canary.token[:half], canary.token[half:] + " — done."]

    blocked = False
    for chunk in chunks:
        _, is_blocked, _ = buf.feed(chunk)
        if is_blocked:
            blocked = True
            break
    assert blocked  # detected once the full token lands in the rolling window


def test_streaming_buffer_passes_clean_multi_chunk_text_through_losslessly():
    canary = generate_canary()
    buf = StreamingDLPBuffer(canary, buffer_chars=16, hold_back=True)
    parts = ["Once upon ", "a time, ", "there was a ", "helpful assistant."]
    out = ""
    for part in parts:
        text, blocked, _ = buf.feed(part)
        assert not blocked
        out += text
    tail, blocked, _ = buf.flush()
    assert not blocked
    out += tail
    assert out == "".join(parts)
