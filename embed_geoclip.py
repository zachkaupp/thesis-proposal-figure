"""GeoCLIP location embeddings per H3 cell, from the pretrained location encoder applied to each cell center.

    uv run python embed_geoclip.py
"""
import pandas as pd
import torch
from geoclip import LocationEncoder

from config import GRID_PATH, embedding_path


def main():
    grid = pd.read_parquet(GRID_PATH, columns=["lat", "lon"])
    encoder = LocationEncoder()  # loads the pretrained weights shipped with the package
    encoder.eval()
    gps = torch.tensor(grid[["lat", "lon"]].to_numpy(), dtype=torch.float32)  # GeoCLIP takes (lat, lon)
    with torch.no_grad():
        features = torch.cat([encoder(batch) for batch in gps.split(1024)])
    embeddings = pd.DataFrame(features.numpy(), index=grid.index,
                              columns=[f"geoclip_{i}" for i in range(features.shape[1])])
    embeddings.to_parquet(embedding_path("geoclip"))
    print(f"Saved {embeddings.shape} to {embedding_path('geoclip')}")


if __name__ == "__main__":
    main()
