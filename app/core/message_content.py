from __future__ import annotations

from copy import deepcopy
from typing import Any


def extract_text_parts(content: Any) -> list[str]:
    """Collect text from OpenAI-style string or multimodal content."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
                # Handle nested provider-specific content structures safely.
                for value in part.values():
                    if isinstance(value, (dict, list)):
                        texts.extend(extract_text_parts(value))
        return texts
    if isinstance(content, dict):
        text = content.get("text")
        return [text] if isinstance(text, str) else []
    return []


def sanitize_content(content: Any, sanitizer) -> tuple[Any, list[str]]:
    """Sanitize every textual content part and return (copy, scanned_text)."""
    if isinstance(content, str):
        sanitized = sanitizer(content)
        return sanitized, [sanitized]
    if isinstance(content, list):
        output = deepcopy(content)
        scanned: list[str] = []
        for part in output:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    part["text"] = sanitizer(part["text"])
                    scanned.append(part["text"])
                for key, value in list(part.items()):
                    if isinstance(value, (dict, list)):
                        sanitized_value, nested = sanitize_content(value, sanitizer)
                        part[key] = sanitized_value
                        scanned.extend(nested)
        return output, scanned
    return content, []
