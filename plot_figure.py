"""Three-panel map of cosine similarity to a reference cell (JFK) for AlphaEarth, Hex2Vec and GeoCLIP.

    uv run python plot_figure.py

Similarity is shown as a percentile within each panel, since raw cosine ranges differ between embeddings.
Writes figures/nyc_similarity.pdf and figures/nyc_similarity.png.
"""
import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pyproj import Transformer

from config import (FIGURE_DIR, GRID_PATH, H3_RES, LAND_PATH, PLOT_CRS, REFERENCE_LATLON,
                    embedding_path, square_bounds)

PANELS = [("alphaearth", "AlphaEarth", "satellite"),
          ("hex2vec", "Hex2Vec", "OpenStreetMap"),
          ("geoclip", "GeoCLIP", "geotagged photos")]

# magma reversed: pale yellow (low) to near-black purple (high), perceptually uniform and colorblind-safe
CMAP = plt.get_cmap("magma_r")
WATER = "#e4e3df"
LAND = "#f7f6f3"
NO_DATA = "#bdbcb8"
MARKER = "#111111"


def similarity_percentile(embeddings, reference):
    unit = embeddings.to_numpy() / np.linalg.norm(embeddings.to_numpy(), axis=1, keepdims=True)
    cosine = pd.Series(unit @ unit[embeddings.index.get_loc(reference)], index=embeddings.index)
    return cosine.rank(pct=True)


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    grid = gpd.read_parquet(GRID_PATH).to_crs(PLOT_CRS)
    land = gpd.read_parquet(LAND_PATH).to_crs(PLOT_CRS)

    reference = h3.latlng_to_cell(*REFERENCE_LATLON, H3_RES)
    ref_x, ref_y = Transformer.from_crs("EPSG:4326", PLOT_CRS, always_xy=True).transform(
        *h3.cell_to_latlng(reference)[::-1])
    left, bottom, right, top = square_bounds()

    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.5, 3.2), layout="constrained")
    norm = Normalize(0, 1)
    for ax, (key, name, source) in zip(axes, PANELS):
        embeddings = pd.read_parquet(embedding_path(key))
        if reference not in embeddings.index:
            raise ValueError(f"reference cell {reference} has no {name} embedding")
        values = grid.join(similarity_percentile(embeddings, reference).rename("value"))
        missing = values["value"].isna()
        if missing.any():
            print(f"{name}: {missing.sum()} cells without an embedding, drawn in gray")

        ax.set_facecolor(WATER)
        land.plot(ax=ax, color=LAND, linewidth=0)
        values[missing].plot(ax=ax, color=NO_DATA, linewidth=0)
        values[~missing].plot(ax=ax, column="value", cmap=CMAP, norm=norm, edgecolor="face", linewidth=0.15)
        # white star with a dark edge, since the most similar cells around the reference are near-black
        ax.plot(ref_x, ref_y, marker="*", markersize=10, color="white", markeredgecolor=MARKER, markeredgewidth=0.8)

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")  # geopandas labels the axes from the CRS (easting/northing)
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{name}\n({source})", fontsize=9)

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=axes, orientation="horizontal",
                            shrink=0.45, aspect=30, pad=0.02)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "25", "50", "75", "100"])
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Cosine similarity to reference cell (percentile)", fontsize=8)

    for suffix, kwargs in [("pdf", {}), ("png", {"dpi": 300})]:
        path = FIGURE_DIR / f"nyc_similarity.{suffix}"
        fig.savefig(path, **kwargs)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
