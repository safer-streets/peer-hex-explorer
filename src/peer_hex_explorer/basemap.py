"""Static close-up maps of single cells: the hex outlined over an OSM basemap, like the PDFs' contextily panels.

Drawn server-side into a PNG rather than as a pydeck map per panel. Each deck.gl map holds a WebGL context, browsers
cap those at ~16 per page, and a target plus 20 peers plus the overview map would go past that and start dropping
maps.
"""

import io
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# OpenStreetMap's standard tiles. (Carto's raster basemaps, which match pydeck's style, now need an API key.) The OSM
# tile policy allows light use like this -- an identifying User-Agent, tiles cached, at most 2 connections -- but not
# heavy traffic: a busy deployment needs a commercial or self-hosted tile source here.
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_PX = 256
ATTRIBUTION = "© OpenStreetMap contributors"
USER_AGENT = "peer-hex-explorer/0.1 (safer-streets research prototype)"
MAX_CONNECTIONS = 2

# z17 at UK latitudes is ~0.7 m/px, so a cell (404m corner to corner) is ~560px across: an 800px window shows it with
# about a third of a cell of surroundings on every side, from 9-16 tiles, and stays sharp at panel width.
ZOOM = 17
WINDOW_PX = 800

_FONT = ImageFont.load_default(size=14)


def _to_pixels(lon: np.ndarray, lat: np.ndarray, zoom: int = ZOOM) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 -> global web-mercator pixel coordinates at `zoom`."""
    scale = TILE_PX * 2**zoom
    x = (np.asarray(lon) + 180.0) / 360.0 * scale
    lat_rad = np.radians(lat)
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / math.pi) / 2.0 * scale
    return x, y


@lru_cache(maxsize=2048)
def _tile(x: int, y: int, zoom: int = ZOOM) -> Image.Image | None:
    """One basemap tile, or None if it can't be fetched (the panel then draws on a blank ground)."""
    request = urllib.request.Request(TILE_URL.format(z=zoom, x=x, y=y), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")
    except OSError:
        return None


def cell_map_image(outline: list[list[float]], colour: str, width: float = 6) -> bytes:
    """JPEG of the cell `outline` ([[lon, lat], ...], closed) in `colour`, centred over the basemap."""
    lon, lat = np.asarray(outline).T
    px, py = _to_pixels(lon, lat)
    cx, cy = px[:-1].mean(), py[:-1].mean()
    left, top = round(cx - WINDOW_PX / 2), round(cy - WINDOW_PX / 2)

    tiles = [
        (tx, ty)
        for tx in range(left // TILE_PX, (left + WINDOW_PX) // TILE_PX + 1)
        for ty in range(top // TILE_PX, (top + WINDOW_PX) // TILE_PX + 1)
    ]
    with ThreadPoolExecutor(max_workers=MAX_CONNECTIONS) as pool:
        images = list(pool.map(lambda t: _tile(*t), tiles))

    canvas = Image.new("RGB", (WINDOW_PX, WINDOW_PX), (242, 239, 233))
    for (tx, ty), image in zip(tiles, images, strict=True):
        if image is not None:
            canvas.paste(image, (tx * TILE_PX - left, ty * TILE_PX - top))

    vertices = list(zip((px - left).tolist(), (py - top).tolist(), strict=True))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rgb = tuple(int(colour.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    draw.polygon(vertices, fill=(*rgb, 40))
    draw.line(vertices, fill=(*rgb, 255), width=round(width), joint="curve")
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)

    draw = ImageDraw.Draw(canvas)
    text_w = draw.textlength(ATTRIBUTION, font=_FONT)
    draw.rectangle((WINDOW_PX - text_w - 8, WINDOW_PX - 21, WINDOW_PX, WINDOW_PX), fill=(255, 255, 255, 200))
    draw.text((WINDOW_PX - text_w - 4, WINDOW_PX - 19), ATTRIBUTION, fill=(80, 80, 80), font=_FONT)

    # JPEG, not PNG: a basemap is photographic enough that PNG runs ~4x larger, and up to 21 of these go over the
    # websocket at once
    buffer = io.BytesIO()
    canvas.convert("RGB").save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()
