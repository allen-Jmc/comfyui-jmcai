from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import uuid4

import folder_paths
from aiohttp import web
from server import PromptServer

PLUGIN_VERSION = "1.2.0"
UPLOADABLE_MEDIA_KINDS = ("audio", "video", "file")
SANITIZE_FILENAME_RE = re.compile(r"[^\w.-]+")


def _sanitize_filename(file_name: str) -> str:
    normalized = os.path.basename(file_name or "").strip()
    normalized = SANITIZE_FILENAME_RE.sub("-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or "upload.bin"


def _build_target_filename(media_kind: str, original_name: str) -> str:
    sanitized_name = _sanitize_filename(original_name)
    suffix = Path(sanitized_name).suffix
    return f"jmcai_{media_kind}_{uuid4().hex}{suffix}"


async def _read_uploaded_file(request: web.Request) -> tuple[str, bytes]:
    reader = await request.multipart()
    while True:
        field = await reader.next()
        if field is None:
            break

        if field.name != "file":
            continue

        file_name = field.filename or "upload.bin"
        content = await field.read(decode=False)
        return file_name, content

    raise web.HTTPBadRequest(reason="Missing multipart field: file")


def _save_input_file(media_kind: str, original_name: str, content: bytes) -> str:
    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)
    target_name = _build_target_filename(media_kind, original_name)
    target_path = os.path.join(input_dir, target_name)
    with open(target_path, "wb") as file:
        file.write(content)
    return target_name


@PromptServer.instance.routes.get("/jmcai/capabilities")
async def get_jmcai_capabilities(_request: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "data": {
                "provider": "jmcai-plugin",
                "version": PLUGIN_VERSION,
                "uploads": list(UPLOADABLE_MEDIA_KINDS),
            },
        }
    )


@PromptServer.instance.routes.post("/jmcai/upload/{media_kind}")
async def upload_jmcai_asset(request: web.Request) -> web.Response:
    media_kind = str(request.match_info.get("media_kind", "")).strip().lower()
    if media_kind not in UPLOADABLE_MEDIA_KINDS:
        raise web.HTTPNotFound()

    original_name, content = await _read_uploaded_file(request)
    stored_name = _save_input_file(media_kind, original_name, content)

    return web.json_response(
        {
            "ok": True,
            "data": {
                "filename": stored_name,
                "mediaKind": media_kind,
            },
        }
    )
