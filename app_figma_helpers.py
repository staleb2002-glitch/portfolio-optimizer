import os
import requests
from typing import Dict, Any, List

HEADERS_TEMPLATE = {"X-Figma-Token": None}

class FigmaError(RuntimeError):
    pass


def figma_get_file(file_key: str, token: str) -> Dict[str, Any]:
    """Return the full Figma file JSON metadata."""
    if not token:
        raise FigmaError("FIGMA token required")
    url = f"https://api.figma.com/v1/files/{file_key}"
    headers = {"X-Figma-Token": token}
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        raise FigmaError(f"Figma file fetch failed: {r.status_code} {r.text}")
    return r.json()


def figma_get_images(file_key: str, node_ids: str, token: str, scale: int = 2, fmt: str = "png") -> Dict[str, str]:
    """Return a dict mapping node id -> image URL for provided node ids (comma-separated)."""
    if not token:
        raise FigmaError("FIGMA token required")
    url = f"https://api.figma.com/v1/images/{file_key}"
    headers = {"X-Figma-Token": token}
    params = {"ids": node_ids, "scale": scale, "format": fmt}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    if r.status_code != 200:
        raise FigmaError(f"Figma images fetch failed: {r.status_code} {r.text}")
    data = r.json()
    return data.get("images", {})


def figma_get_styles(file_key: str, token: str) -> Dict[str, Any]:
    if not token:
        raise FigmaError("FIGMA token required")
    url = f"https://api.figma.com/v1/files/{file_key}/styles"
    headers = {"X-Figma-Token": token}
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        raise FigmaError(f"Figma styles fetch failed: {r.status_code} {r.text}")
    return r.json()


def list_frames_from_file(file_json: Dict[str, Any]) -> List[Dict[str, str]]:
    """Traverse file JSON and return list of frames with {'id': id, 'name': name}.
    This finds nodes of type 'FRAME' in the document tree.
    """
    out = []
    def _walk(node):
        if not isinstance(node, dict):
            return
        t = node.get("type")
        if t == "FRAME":
            out.append({"id": node.get("id"), "name": node.get("name")})
        for ch in node.get("children", []) or []:
            _walk(ch)
    doc = file_json.get("document")
    if doc:
        _walk(doc)
    return out


if __name__ == "__main__":
    print("app_figma_helpers loaded")
