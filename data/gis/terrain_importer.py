"""GIS terrain importer for E11.2 — real-world map ingestion.

Converts SRTM elevation rasters (GeoTIFF) and OpenStreetMap (OSM)
road/forest/town layers into :class:`~envs.sim.terrain.TerrainMap` objects
for use in historical battle scenarios.

Architecture
------------
:class:`BattleSiteBounds`
    Named bounding-boxes (``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``)
    for the four supported historical sites: Waterloo, Austerlitz, Borodino,
    and Salamanca.  Also provides a synthetic elevation profile that
    approximates the real terrain shape for each site.

:class:`SRTMImporter`
    Reads a single-band GeoTIFF elevation raster cropped to a bounding box.
    Requires ``rasterio`` for real GeoTIFF files; when ``rasterio`` is
    unavailable it falls back to the site's built-in synthetic elevation
    profile so that the rest of the pipeline can be tested without external
    dependencies.

:class:`OSMLayerImporter`
    Parses an OSM XML file for road, forest, and settlement (town) polygon
    features within the bounding box and rasterises each into a binary mask
    at the requested grid resolution.  Requires only the Python standard
    library (``xml.etree.ElementTree``); advanced polygon rasterisation via
    ``shapely`` is used when available but is never required.

:class:`GISTerrainBuilder`
    High-level façade that calls :class:`SRTMImporter` and
    :class:`OSMLayerImporter` and combines their outputs into a single
    :class:`~envs.sim.terrain.TerrainMap`.  Cover values are computed as the
    union of the forest mask and a fraction of the settlement mask.

Typical usage
-------------
Without external data (synthetic fallback)::

    from data.gis.terrain_importer import GISTerrainBuilder, BattleSiteBounds

    builder = GISTerrainBuilder(site="waterloo", rows=40, cols=40)
    terrain = builder.build()
    print(terrain.width, terrain.height, terrain.elevation.shape)

With a real SRTM GeoTIFF and OSM XML file::

    builder = GISTerrainBuilder(
        site="waterloo",
        rows=40,
        cols=40,
        srtm_path="/data/srtm/N50E004.tif",
        osm_path="/data/osm/waterloo.osm",
    )
    terrain = builder.build()
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Attempt to import optional heavy-weight GIS libraries.
# ---------------------------------------------------------------------------

try:
    import rasterio  # type: ignore
    import rasterio.windows  # type: ignore
    from rasterio.transform import from_bounds as _rio_from_bounds  # type: ignore
    _HAVE_RASTERIO = True
except ImportError:  # pragma: no cover
    _HAVE_RASTERIO = False

try:
    from shapely.geometry import box as _shapely_box  # type: ignore
    from shapely.geometry import Polygon as _ShapelyPolygon  # type: ignore
    _HAVE_SHAPELY = True
except ImportError:  # pragma: no cover
    _HAVE_SHAPELY = False

# Add project root to sys.path so we can import envs regardless of how the
# module is invoked.
_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from envs.sim.terrain import TerrainMap  # noqa: E402 (import after sys.path update)


# ---------------------------------------------------------------------------
# BattleSiteBounds
# ---------------------------------------------------------------------------

# Approximate width/height in metres for each battle site box.
# 1 degree latitude ≈ 111 km; 1 degree longitude ≈ 111 km × cos(lat).
_DEG_LAT_M = 111_320.0  # metres per degree latitude


def _lon_deg_to_m(lat_deg: float) -> float:
    """Metres per degree of longitude at *lat_deg* degrees latitude."""
    return _DEG_LAT_M * math.cos(math.radians(lat_deg))


@dataclass(frozen=True)
class BattleSiteBounds:
    """Geographic bounding box for a historical battle site.

    Attributes
    ----------
    site_id:
        Short slug identifier, e.g. ``"waterloo"``.
    lat_min, lat_max:
        Latitude range (decimal degrees, WGS-84).
    lon_min, lon_max:
        Longitude range (decimal degrees, WGS-84).
    description:
        Human-readable label.

    Notes
    -----
    Each site covers roughly 10 km × 8 km.  The synthetic elevation
    profiles are parameterised from published topographic information:

    * **Waterloo** — Ridge of Mont-Saint-Jean (elevation ≈ 100–150 m ASL)
      running E–W across the northern part of the field, with lower ground
      toward the French position to the south.
    * **Austerlitz** — Pratzen Heights (elevation ≈ 280–310 m ASL) forming
      a dominant plateau in the centre of the field.
    * **Borodino** — Gentle ridge from Gorky to Utitsa with the Raevsky
      Redoubt on a modest spur; elevation range ≈ 200–240 m ASL.
    * **Salamanca** — Greater and Lesser Arapiles ridges (≈ 800–850 m ASL)
      flanking the Tormes valley.
    """

    site_id: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    description: str = ""

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    @property
    def width_m(self) -> float:
        """Approximate width (east–west) in metres."""
        mid_lat = (self.lat_min + self.lat_max) / 2.0
        return abs(self.lon_max - self.lon_min) * _lon_deg_to_m(mid_lat)

    @property
    def height_m(self) -> float:
        """Approximate height (north–south) in metres."""
        return abs(self.lat_max - self.lat_min) * _DEG_LAT_M

    # ------------------------------------------------------------------
    # Synthetic elevation factory
    # ------------------------------------------------------------------

    def synthetic_elevation(self, rows: int, cols: int) -> np.ndarray:
        """Generate a synthetic elevation grid approximating this site.

        The profile is constructed from Gaussian blobs whose positions and
        sizes are calibrated against published topographic descriptions.
        Values are normalised to ``[0, 1]``.

        Parameters
        ----------
        rows, cols:
            Output grid resolution.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(rows, cols)`` with values in
            ``[0, 1]``.
        """
        return _SYNTHETIC_ELEVATIONS[self.site_id](rows, cols)

    def synthetic_cover(self, rows: int, cols: int) -> np.ndarray:
        """Generate a synthetic cover (forest/settlement) grid for this site.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(rows, cols)`` with values in
            ``[0, 1]``.
        """
        return _SYNTHETIC_COVERS[self.site_id](rows, cols)


# ---------------------------------------------------------------------------
# Pre-defined battle site bounding boxes
# ---------------------------------------------------------------------------

BATTLE_SITES: Dict[str, BattleSiteBounds] = {
    "waterloo": BattleSiteBounds(
        site_id="waterloo",
        lat_min=50.66,
        lat_max=50.74,
        lon_min=4.36,
        lon_max=4.46,
        description="Battle of Waterloo (1815-06-18), Belgium",
    ),
    "austerlitz": BattleSiteBounds(
        site_id="austerlitz",
        lat_min=49.09,
        lat_max=49.17,
        lon_min=16.73,
        lon_max=16.85,
        description="Battle of Austerlitz (1805-12-02), Czech Republic",
    ),
    "borodino": BattleSiteBounds(
        site_id="borodino",
        lat_min=55.49,
        lat_max=55.55,
        lon_min=35.79,
        lon_max=35.90,
        description="Battle of Borodino (1812-09-07), Russia",
    ),
    "salamanca": BattleSiteBounds(
        site_id="salamanca",
        lat_min=40.90,
        lat_max=40.98,
        lon_min=-5.55,
        lon_max=-5.42,
        description="Battle of Salamanca (1812-07-22), Spain",
    ),
}


# ---------------------------------------------------------------------------
# Synthetic terrain generators
# ---------------------------------------------------------------------------

def _gaussian_blob(
    rows: int,
    cols: int,
    cy: float,
    cx: float,
    sigma: float,
    scale: float = 1.0,
) -> np.ndarray:
    """Return a 2-D Gaussian blob in normalised [0,1]×[0,1] coordinates."""
    ys = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    xs = np.linspace(0.0, 1.0, cols, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    return (
        scale
        * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2.0 * sigma ** 2))
    ).astype(np.float32)


def _normalise(arr: np.ndarray) -> np.ndarray:
    mx = float(arr.max())
    if mx > 0.0:
        return (arr / mx).astype(np.float32)
    return arr.astype(np.float32)


def _waterloo_elevation(rows: int, cols: int) -> np.ndarray:
    """Mont-Saint-Jean ridge (E–W) across y ≈ 0.55–0.65."""
    elev = _gaussian_blob(rows, cols, cy=0.60, cx=0.50, sigma=0.18, scale=1.0)
    elev += _gaussian_blob(rows, cols, cy=0.58, cx=0.35, sigma=0.10, scale=0.6)
    # La Belle Alliance ridge south of centre
    elev += _gaussian_blob(rows, cols, cy=0.35, cx=0.50, sigma=0.12, scale=0.4)
    return _normalise(elev)


def _waterloo_cover(rows: int, cols: int) -> np.ndarray:
    """Hougoumont orchard (SW), La Haye Sainte environs, Bois de Paris (SE)."""
    cov = _gaussian_blob(rows, cols, cy=0.60, cx=0.25, sigma=0.06, scale=0.8)
    cov += _gaussian_blob(rows, cols, cy=0.62, cx=0.50, sigma=0.04, scale=0.6)
    cov += _gaussian_blob(rows, cols, cy=0.40, cx=0.78, sigma=0.07, scale=0.9)
    return np.clip(cov, 0.0, 1.0).astype(np.float32)


def _austerlitz_elevation(rows: int, cols: int) -> np.ndarray:
    """Pratzen Heights plateau dominating the centre."""
    elev = _gaussian_blob(rows, cols, cy=0.55, cx=0.55, sigma=0.22, scale=1.0)
    elev += _gaussian_blob(rows, cols, cy=0.65, cx=0.35, sigma=0.08, scale=0.5)
    # Santon Hill (NW corner)
    elev += _gaussian_blob(rows, cols, cy=0.70, cx=0.20, sigma=0.06, scale=0.7)
    # Zuran Hill (W)
    elev += _gaussian_blob(rows, cols, cy=0.55, cx=0.25, sigma=0.07, scale=0.6)
    return _normalise(elev)


def _austerlitz_cover(rows: int, cols: int) -> np.ndarray:
    """Telnitz village area and Sokolnitz woods."""
    cov = _gaussian_blob(rows, cols, cy=0.25, cx=0.65, sigma=0.05, scale=0.7)
    cov += _gaussian_blob(rows, cols, cy=0.30, cx=0.70, sigma=0.06, scale=0.6)
    return np.clip(cov, 0.0, 1.0).astype(np.float32)


def _borodino_elevation(rows: int, cols: int) -> np.ndarray:
    """Gentle ridge from Gorky to Utitsa; Raevsky Redoubt spur."""
    elev = _gaussian_blob(rows, cols, cy=0.55, cx=0.65, sigma=0.20, scale=1.0)
    # Raevsky Redoubt spur
    elev += _gaussian_blob(rows, cols, cy=0.55, cx=0.70, sigma=0.07, scale=0.6)
    # Southern Fleches sector
    elev += _gaussian_blob(rows, cols, cy=0.35, cx=0.60, sigma=0.10, scale=0.4)
    return _normalise(elev)


def _borodino_cover(rows: int, cols: int) -> np.ndarray:
    """Utitsa forest and river Kolocha banks."""
    cov = _gaussian_blob(rows, cols, cy=0.25, cx=0.60, sigma=0.08, scale=0.9)
    cov += _gaussian_blob(rows, cols, cy=0.55, cx=0.45, sigma=0.06, scale=0.5)
    cov += _gaussian_blob(rows, cols, cy=0.65, cx=0.35, sigma=0.05, scale=0.6)
    return np.clip(cov, 0.0, 1.0).astype(np.float32)


def _salamanca_elevation(rows: int, cols: int) -> np.ndarray:
    """Greater and Lesser Arapiles ridges flanking the Tormes valley."""
    # Greater Arapil (E)
    elev = _gaussian_blob(rows, cols, cy=0.45, cx=0.65, sigma=0.07, scale=1.0)
    # Lesser Arapil (W)
    elev += _gaussian_blob(rows, cols, cy=0.50, cx=0.35, sigma=0.06, scale=0.8)
    # Tormes valley floor (low centre)
    elev += _gaussian_blob(rows, cols, cy=0.75, cx=0.50, sigma=0.15, scale=0.2)
    return _normalise(elev)


def _salamanca_cover(rows: int, cols: int) -> np.ndarray:
    """Sparse woodland patches around the Arapiles."""
    cov = _gaussian_blob(rows, cols, cy=0.55, cx=0.50, sigma=0.10, scale=0.5)
    cov += _gaussian_blob(rows, cols, cy=0.35, cx=0.55, sigma=0.06, scale=0.4)
    return np.clip(cov, 0.0, 1.0).astype(np.float32)


_SYNTHETIC_ELEVATIONS = {
    "waterloo": _waterloo_elevation,
    "austerlitz": _austerlitz_elevation,
    "borodino": _borodino_elevation,
    "salamanca": _salamanca_elevation,
}

_SYNTHETIC_COVERS = {
    "waterloo": _waterloo_cover,
    "austerlitz": _austerlitz_cover,
    "borodino": _borodino_cover,
    "salamanca": _salamanca_cover,
}


# ---------------------------------------------------------------------------
# SRTMImporter
# ---------------------------------------------------------------------------


class SRTMImporter:
    """Import elevation data from an SRTM GeoTIFF file.

    Parameters
    ----------
    srtm_path:
        Path to a GeoTIFF raster containing elevation in metres.
        If ``None`` or if ``rasterio`` is unavailable, the synthetic
        fallback for *site* is used instead.
    site:
        One of the keys in :data:`BATTLE_SITES`.  Used to select the
        bounding box and synthetic fallback.
    rows, cols:
        Target grid resolution of the output elevation array.

    Raises
    ------
    KeyError
        If *site* is not one of the known :data:`BATTLE_SITES` keys.
    """

    def __init__(
        self,
        site: str,
        rows: int = 40,
        cols: int = 40,
        srtm_path: Optional[str | Path] = None,
    ) -> None:
        if site not in BATTLE_SITES:
            raise KeyError(
                f"Unknown battle site {site!r}. "
                f"Known sites: {sorted(BATTLE_SITES)}"
            )
        self.site = site
        self.bounds = BATTLE_SITES[site]
        self.rows = int(rows)
        self.cols = int(cols)
        self.srtm_path = Path(srtm_path) if srtm_path is not None else None

    def load(self) -> np.ndarray:
        """Return a ``(rows, cols)`` float32 elevation array in metres.

        Tries to read *srtm_path* with ``rasterio`` first; falls back to the
        synthetic profile when the file is absent or ``rasterio`` is not
        installed.  The returned values are *not* normalised so that the
        caller can choose whether to normalise against the whole scene.
        """
        if (
            self.srtm_path is not None
            and self.srtm_path.exists()
            and _HAVE_RASTERIO
        ):
            return self._load_rasterio()
        return self._synthetic_metres()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_rasterio(self) -> np.ndarray:
        """Read and crop a GeoTIFF using rasterio, then resample."""
        import rasterio  # local import; already guarded by _HAVE_RASTERIO
        from rasterio.enums import Resampling  # type: ignore

        b = self.bounds
        with rasterio.open(self.srtm_path) as src:
            window = rasterio.windows.from_bounds(
                b.lon_min, b.lat_min, b.lon_max, b.lat_max,
                transform=src.transform,
            )
            # Read band 1, resampling to the requested grid size
            data = src.read(
                1,
                window=window,
                out_shape=(self.rows, self.cols),
                resampling=Resampling.bilinear,
            )
        return data.astype(np.float32)

    def _synthetic_metres(self) -> np.ndarray:
        """Generate synthetic elevation in metres using the site profile."""
        norm = self.bounds.synthetic_elevation(self.rows, self.cols)
        # Scale to a plausible elevation range (50–300 m ASL depending on site)
        _ELEV_RANGES = {
            "waterloo": (80.0, 160.0),
            "austerlitz": (220.0, 320.0),
            "borodino": (190.0, 240.0),
            "salamanca": (780.0, 870.0),
        }
        lo, hi = _ELEV_RANGES.get(self.site, (0.0, 200.0))
        return (lo + norm * (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# OSMLayerImporter
# ---------------------------------------------------------------------------

# OSM tags that represent road features
_ROAD_HIGHWAY_VALUES = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "track", "path",
})

# OSM tags (natural/landuse) that represent forest/vegetation
_FOREST_TAG_VALUES = frozenset({
    ("natural", "wood"),
    ("landuse", "forest"),
    ("natural", "scrub"),
    ("landuse", "meadow"),
})

# OSM tags for settlements
_SETTLEMENT_TAG_VALUES = frozenset({
    ("place", "town"),
    ("place", "village"),
    ("place", "hamlet"),
    ("landuse", "residential"),
})


@dataclass
class OSMLayers:
    """Rasterised OSM layer masks at the same grid resolution.

    Attributes
    ----------
    road_mask:
        Binary float32 mask — ``1.0`` where a road segment passes through the
        cell, ``0.0`` elsewhere.
    forest_mask:
        Float32 mask — fraction of the cell covered by forest/vegetation
        polygons (approximate when ``shapely`` is absent).
    settlement_mask:
        Float32 mask — fraction of the cell covered by settlement polygons.
    """

    road_mask: np.ndarray
    forest_mask: np.ndarray
    settlement_mask: np.ndarray


class OSMLayerImporter:
    """Parse an OSM XML file and rasterise road/forest/settlement layers.

    Parameters
    ----------
    osm_path:
        Path to an OSM XML (``.osm``) file.  If ``None`` or absent, a
        synthetic mask is generated from the site's known geography.
    site:
        One of the keys in :data:`BATTLE_SITES`.
    rows, cols:
        Target grid resolution.
    """

    def __init__(
        self,
        site: str,
        rows: int = 40,
        cols: int = 40,
        osm_path: Optional[str | Path] = None,
    ) -> None:
        if site not in BATTLE_SITES:
            raise KeyError(
                f"Unknown battle site {site!r}. "
                f"Known sites: {sorted(BATTLE_SITES)}"
            )
        self.site = site
        self.bounds = BATTLE_SITES[site]
        self.rows = int(rows)
        self.cols = int(cols)
        self.osm_path = Path(osm_path) if osm_path is not None else None

    def load(self) -> OSMLayers:
        """Parse the OSM file (or generate synthetic masks) and return layers."""
        if self.osm_path is not None and self.osm_path.exists():
            return self._parse_osm()
        return self._synthetic_layers()

    # ------------------------------------------------------------------
    # OSM XML parser
    # ------------------------------------------------------------------

    def _parse_osm(self) -> OSMLayers:
        """Parse an OSM XML file and rasterise node/way features."""
        tree = ET.parse(self.osm_path)
        root = tree.getroot()

        # Index node coordinates by ID
        nodes: Dict[str, Tuple[float, float]] = {}
        for node in root.iter("node"):
            nid = node.get("id", "")
            lat = node.get("lat")
            lon = node.get("lon")
            if lat is not None and lon is not None:
                nodes[nid] = (float(lat), float(lon))

        road_mask = np.zeros((self.rows, self.cols), dtype=np.float32)
        forest_mask = np.zeros((self.rows, self.cols), dtype=np.float32)
        settlement_mask = np.zeros((self.rows, self.cols), dtype=np.float32)

        for way in root.iter("way"):
            tags = {
                tag.get("k", ""): tag.get("v", "")
                for tag in way.iter("tag")
            }
            ndrefs = [nd.get("ref", "") for nd in way.iter("nd")]

            # Classify
            is_road = tags.get("highway", "") in _ROAD_HIGHWAY_VALUES
            is_forest = any((k, v) in _FOREST_TAG_VALUES for k, v in tags.items())
            is_settlement = any(
                (k, v) in _SETTLEMENT_TAG_VALUES for k, v in tags.items()
            )

            if not (is_road or is_forest or is_settlement):
                continue

            # Collect lat/lon of the way's nodes
            coords = [nodes[r] for r in ndrefs if r in nodes]
            if len(coords) < 2:
                continue

            if is_road:
                self._rasterise_linestring(road_mask, coords)
            if is_forest:
                self._rasterise_polygon(forest_mask, coords)
            if is_settlement:
                self._rasterise_polygon(settlement_mask, coords)

        return OSMLayers(
            road_mask=np.clip(road_mask, 0.0, 1.0),
            forest_mask=np.clip(forest_mask, 0.0, 1.0),
            settlement_mask=np.clip(settlement_mask, 0.0, 1.0),
        )

    def _latlon_to_cell(
        self, lat: float, lon: float
    ) -> Tuple[int, int]:
        """Convert a (lat, lon) coordinate to a (row, col) grid index."""
        b = self.bounds
        col_frac = (lon - b.lon_min) / (b.lon_max - b.lon_min)
        row_frac = (lat - b.lat_min) / (b.lat_max - b.lat_min)
        col = int(np.clip(col_frac * self.cols, 0, self.cols - 1))
        row = int(np.clip(row_frac * self.rows, 0, self.rows - 1))
        return row, col

    def _rasterise_linestring(
        self,
        mask: np.ndarray,
        coords: List[Tuple[float, float]],
    ) -> None:
        """Mark grid cells traversed by a polyline as 1."""
        for (lat0, lon0), (lat1, lon1) in zip(coords[:-1], coords[1:]):
            r0, c0 = self._latlon_to_cell(lat0, lon0)
            r1, c1 = self._latlon_to_cell(lat1, lon1)
            # Bresenham-style cell traversal
            for r, c in _bresenham(r0, c0, r1, c1):
                mask[r, c] = 1.0

    def _rasterise_polygon(
        self,
        mask: np.ndarray,
        coords: List[Tuple[float, float]],
    ) -> None:
        """Fill grid cells inside a polygon (approximate scanline fill)."""
        if len(coords) < 3:
            return
        # Convert to grid coords
        grid_coords = [self._latlon_to_cell(lat, lon) for lat, lon in coords]
        rows_arr = [r for r, _ in grid_coords]
        cols_arr = [c for _, c in grid_coords]
        r_min = max(0, min(rows_arr))
        r_max = min(self.rows - 1, max(rows_arr))
        c_min = max(0, min(cols_arr))
        c_max = min(self.cols - 1, max(cols_arr))
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if _point_in_polygon(r, c, grid_coords):
                    mask[r, c] = 1.0

    # ------------------------------------------------------------------
    # Synthetic layer generator
    # ------------------------------------------------------------------

    def _synthetic_layers(self) -> OSMLayers:
        """Return synthetic road/forest/settlement masks for the site."""
        cover = self.bounds.synthetic_cover(self.rows, self.cols)
        # Roads: thin E–W and N–S corridors
        road_mask = self._synthetic_road_mask()
        # Forest uses the cover profile directly
        forest_mask = (cover > 0.3).astype(np.float32) * cover
        # Settlements: small blobs at historically known village locations
        settlement_mask = self._synthetic_settlement_mask()
        return OSMLayers(
            road_mask=road_mask,
            forest_mask=forest_mask,
            settlement_mask=settlement_mask,
        )

    def _synthetic_road_mask(self) -> np.ndarray:
        """Generate synthetic road corridors for the site."""
        mask = np.zeros((self.rows, self.cols), dtype=np.float32)
        # Main road along row at y ≈ 0.55 (typical centre of field)
        mid_row = int(0.55 * self.rows)
        if 0 <= mid_row < self.rows:
            mask[mid_row, :] = 1.0
        # N–S road through col at x ≈ 0.50
        mid_col = int(0.50 * self.cols)
        if 0 <= mid_col < self.cols:
            mask[:, mid_col] = 1.0
        return mask

    def _synthetic_settlement_mask(self) -> np.ndarray:
        """Small settlement blobs at site-specific village locations."""
        mask = np.zeros((self.rows, self.cols), dtype=np.float32)
        _VILLAGE_LOCS: Dict[str, List[Tuple[float, float]]] = {
            "waterloo": [(0.62, 0.50), (0.45, 0.30), (0.45, 0.68)],
            "austerlitz": [(0.50, 0.55), (0.25, 0.62), (0.35, 0.72)],
            "borodino": [(0.55, 0.70), (0.35, 0.55), (0.45, 0.60)],
            "salamanca": [(0.45, 0.50), (0.50, 0.70), (0.50, 0.30)],
        }
        for cy, cx in _VILLAGE_LOCS.get(self.site, []):
            blob = _gaussian_blob(
                self.rows, self.cols, cy=cy, cx=cx, sigma=0.04, scale=1.0
            )
            mask = np.maximum(mask, (blob > 0.5).astype(np.float32))
        return mask


# ---------------------------------------------------------------------------
# Bresenham / point-in-polygon helpers
# ---------------------------------------------------------------------------


def _bresenham(
    r0: int, c0: int, r1: int, c1: int
) -> List[Tuple[int, int]]:
    """Return grid cells on the line from (r0,c0) to (r1,c1) (Bresenham)."""
    cells: List[Tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


_EPSILON = 1e-12  # avoids division-by-zero in _point_in_polygon


def _point_in_polygon(
    r: int, c: int, polygon: List[Tuple[int, int]]
) -> bool:
    """Ray-casting point-in-polygon test on integer grid coordinates."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        ri, ci = polygon[i]
        rj, cj = polygon[j]
        if (ci > c) != (cj > c) and r < (rj - ri) * (c - ci) / (cj - ci + _EPSILON) + ri:
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# GISTerrainBuilder
# ---------------------------------------------------------------------------


class GISTerrainBuilder:
    """High-level façade: build a :class:`~envs.sim.terrain.TerrainMap` from
    SRTM + OSM data (real or synthetic).

    Parameters
    ----------
    site:
        One of the keys in :data:`BATTLE_SITES`.
    rows, cols:
        Grid resolution of the output :class:`~envs.sim.terrain.TerrainMap`.
    srtm_path:
        Optional path to a GeoTIFF elevation raster.  If absent or if
        ``rasterio`` is not installed, the synthetic fallback is used.
    osm_path:
        Optional path to an OSM XML file.  If absent, synthetic layer masks
        are used.
    forest_cover_weight:
        Weight applied to the forest mask when computing the combined cover
        layer.  Must be in ``[0, 1]``.
    settlement_cover_weight:
        Weight applied to the settlement mask.  Must be in ``[0, 1]``.

    Examples
    --------
    Synthetic terrain (no files needed)::

        builder = GISTerrainBuilder(site="waterloo", rows=40, cols=40)
        terrain = builder.build()

    With real data::

        builder = GISTerrainBuilder(
            site="waterloo",
            rows=40,
            cols=40,
            srtm_path="/data/N50E004.tif",
            osm_path="/data/waterloo.osm",
        )
        terrain = builder.build()
    """

    def __init__(
        self,
        site: str,
        rows: int = 40,
        cols: int = 40,
        srtm_path: Optional[str | Path] = None,
        osm_path: Optional[str | Path] = None,
        forest_cover_weight: float = 1.0,
        settlement_cover_weight: float = 0.5,
    ) -> None:
        if site not in BATTLE_SITES:
            raise KeyError(
                f"Unknown battle site {site!r}. "
                f"Known sites: {sorted(BATTLE_SITES)}"
            )
        self.site = site
        self.bounds = BATTLE_SITES[site]
        self.rows = int(rows)
        self.cols = int(cols)
        self.srtm_importer = SRTMImporter(
            site=site, rows=self.rows, cols=self.cols, srtm_path=srtm_path
        )
        self.osm_importer = OSMLayerImporter(
            site=site, rows=self.rows, cols=self.cols, osm_path=osm_path
        )
        self.forest_cover_weight = float(forest_cover_weight)
        self.settlement_cover_weight = float(settlement_cover_weight)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> TerrainMap:
        """Build and return a :class:`~envs.sim.terrain.TerrainMap`.

        Steps
        -----
        1. Load elevation from SRTM (real or synthetic).
        2. Normalise elevation to ``[0, 1]``.
        3. Load OSM layers (real or synthetic).
        4. Combine forest and settlement masks into a single cover grid.
        5. Return a :class:`~envs.sim.terrain.TerrainMap` sized to
           ``(bounds.width_m, bounds.height_m)`` in world metres.

        Returns
        -------
        TerrainMap
        """
        elevation_m = self.srtm_importer.load()
        elevation = _normalise_elevation(elevation_m)

        layers = self.osm_importer.load()
        cover = np.clip(
            layers.forest_mask * self.forest_cover_weight
            + layers.settlement_mask * self.settlement_cover_weight,
            0.0,
            1.0,
        ).astype(np.float32)

        return TerrainMap.from_arrays(
            width=self.bounds.width_m,
            height=self.bounds.height_m,
            elevation=elevation,
            cover=cover,
        )

    def build_road_network_data(self) -> np.ndarray:
        """Return the road mask as a float32 grid for road-network integration.

        The mask can be passed to
        :class:`~envs.sim.road_network.RoadNetwork` or used to modulate
        movement speed via the road-speed bonus.

        Returns
        -------
        np.ndarray
            Shape ``(rows, cols)``, values in ``{0.0, 1.0}``.
        """
        layers = self.osm_importer.load()
        return layers.road_mask


# ---------------------------------------------------------------------------
# Elevation normalisation helper
# ---------------------------------------------------------------------------


def _normalise_elevation(elevation_m: np.ndarray) -> np.ndarray:
    """Normalise an elevation array (in metres) to ``[0, 1]``.

    The normalisation preserves *relative* height differences — only the
    minimum elevation within the tile is subtracted before dividing by the
    range.  If the tile is completely flat (min == max) a zero array is
    returned.

    Parameters
    ----------
    elevation_m:
        Raw elevation array in metres (arbitrary ASL offset).

    Returns
    -------
    np.ndarray
        Float32 array with the same shape and values in ``[0, 1]``.
    """
    arr = np.asarray(elevation_m, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    rng = hi - lo
    if rng <= 0.0:
        return np.zeros_like(arr)
    return ((arr - lo) / rng).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: build terrain for all four battle sites
# ---------------------------------------------------------------------------


def build_all_battle_terrains(
    rows: int = 40,
    cols: int = 40,
    data_dir: Optional[str | Path] = None,
) -> Dict[str, TerrainMap]:
    """Build :class:`~envs.sim.terrain.TerrainMap` objects for all four sites.

    Parameters
    ----------
    rows, cols:
        Grid resolution.
    data_dir:
        Optional directory containing ``{site}.tif`` SRTM files and
        ``{site}.osm`` OSM files.  When absent the synthetic fallback is
        used for every site.

    Returns
    -------
    dict[str, TerrainMap]
        Mapping ``site_id → TerrainMap`` for ``"waterloo"``,
        ``"austerlitz"``, ``"borodino"``, and ``"salamanca"``.
    """
    result: Dict[str, TerrainMap] = {}
    base = Path(data_dir) if data_dir is not None else None
    for site in BATTLE_SITES:
        srtm_path = (base / f"{site}.tif") if base is not None else None
        osm_path = (base / f"{site}.osm") if base is not None else None
        builder = GISTerrainBuilder(
            site=site,
            rows=rows,
            cols=cols,
            srtm_path=srtm_path,
            osm_path=osm_path,
        )
        result[site] = builder.build()
    return result
