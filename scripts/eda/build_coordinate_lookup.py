"""
build_coordinate_lookup.py
===========================
Download IBTrACS v04 CSV and build a per-timestep coordinate lookup
for every (storm_name, timestamp) pair in the SETCD dataset.

Usage:
    python build_coordinate_lookup.py \
        --root_dir /kaggle/input/datasets/meheruzannat/cyclone-dataset \
        --years 2006 2007 2008 2009 2010 2011 2012 \
        --output coordinate_lookup.json

The output JSON maps  "storm_name/timestamp" → {"lat": float, "lon": float}
where lat/lon are the IBTrACS best-track positions at that time.
"""

import argparse
import csv
import json
import os
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)


def download_ibtracs(cache_path):
    """Download IBTrACS ALL CSV if not already cached."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"  Using cached IBTrACS CSV: {cache_path} ({size_mb:.1f} MB)")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading IBTrACS ALL CSV (~300 MB)...")
    print(f"  URL: {IBTRACS_URL}")
    urllib.request.urlretrieve(IBTRACS_URL, str(cache_path))
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Downloaded: {cache_path} ({size_mb:.1f} MB)")
    return cache_path


def parse_ibtracs_csv(csv_path):
    """Parse IBTrACS CSV into a dict: {SID: [(iso_time, lat, lon), ...]}"""
    print(f"  Parsing IBTrACS CSV...")
    storm_tracks = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        # Skip the first row (header) and the second row (units)
        reader = csv.reader(f)
        header = next(reader)
        _units = next(reader)  # skip units row
        
        # Find column indices
        try:
            sid_idx = header.index('SID')
            time_idx = header.index('ISO_TIME')
            lat_idx = header.index('LAT')
            lon_idx = header.index('LON')
        except ValueError as e:
            print(f"  ERROR: Could not find required column: {e}")
            print(f"  Available columns: {header[:20]}...")
            raise
        
        count = 0
        for row in reader:
            try:
                sid = row[sid_idx].strip()
                iso_time = row[time_idx].strip()
                lat_str = row[lat_idx].strip()
                lon_str = row[lon_idx].strip()
                
                if not sid or not iso_time or not lat_str or not lon_str:
                    continue
                
                lat = float(lat_str)
                lon = float(lon_str)
                
                # Parse ISO_TIME: "YYYY-MM-DD HH:MM:SS"
                try:
                    dt = datetime.strptime(iso_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                
                storm_tracks[sid].append((dt, lat, lon))
                count += 1
            except (IndexError, ValueError):
                continue
    
    print(f"  ✓ Parsed {count:,} track points across {len(storm_tracks):,} storms")
    return storm_tracks


def scan_dataset_timestamps(root_dir, years):
    """Walk the dataset directory tree and collect (storm_name, timestamp) pairs."""
    root_dir = Path(root_dir)
    entries = []  # list of (storm_name, timestamp_str, year)
    
    for year_folder in years:
        year_path = root_dir / year_folder
        if not year_path.exists():
            continue
        # Handle nested year structure
        nested_path = year_path / year_folder
        if nested_path.exists():
            year_path = nested_path
        
        for cyclone_folder in year_path.iterdir():
            if not cyclone_folder.is_dir():
                continue
            storm_name = cyclone_folder.name
            
            for ts_folder in cyclone_folder.iterdir():
                if not ts_folder.is_dir():
                    continue
                entries.append((storm_name, ts_folder.name, year_folder))
    
    print(f"  Found {len(entries):,} (storm, timestamp) entries across {len(years)} years")
    return entries


def dataset_ts_to_datetime(ts_str):
    """Convert dataset timestamp folder name to datetime.
    Format: 'YYYY-MM-DD HH_MM_SS' → datetime
    """
    try:
        ts_clean = ts_str.split('.')[0]
        return datetime.strptime(ts_clean, "%Y-%m-%d %H_%M_%S")
    except ValueError:
        return None


def find_nearest_track_point(storm_track, target_dt, max_gap_hours=3):
    """Find the IBTrACS track point nearest to target_dt within tolerance."""
    best_point = None
    best_gap = timedelta(hours=max_gap_hours + 1)
    
    for dt, lat, lon in storm_track:
        gap = abs(dt - target_dt)
        if gap < best_gap:
            best_gap = gap
            best_point = (lat, lon)
    
    if best_gap <= timedelta(hours=max_gap_hours):
        return best_point
    return None


def build_lookup(storm_tracks, dataset_entries, max_gap_hours=3):
    """Match each dataset entry to an IBTrACS track point."""
    lookup = {}
    matched = 0
    unmatched_storms = set()
    unmatched_times = 0
    
    for storm_name, ts_str, year in dataset_entries:
        target_dt = dataset_ts_to_datetime(ts_str)
        if target_dt is None:
            continue
        
        # Check if storm SID exists in IBTrACS
        if storm_name not in storm_tracks:
            unmatched_storms.add(storm_name)
            continue
        
        track = storm_tracks[storm_name]
        point = find_nearest_track_point(track, target_dt, max_gap_hours)
        
        if point is not None:
            lat, lon = point
            key = f"{storm_name}/{ts_str}"
            lookup[key] = {"lat": round(lat, 2), "lon": round(lon, 2)}
            matched += 1
        else:
            unmatched_times += 1
    
    total = len(dataset_entries)
    print(f"\n  Match Results:")
    print(f"    Total dataset entries:   {total:,}")
    print(f"    Matched to IBTrACS:      {matched:,} ({100*matched/max(total,1):.1f}%)")
    print(f"    Unmatched (no storm):    {len(unmatched_storms)} unique storms")
    print(f"    Unmatched (time gap):    {unmatched_times:,}")
    
    if unmatched_storms:
        print(f"    Example unmatched SIDs:  {list(unmatched_storms)[:5]}")
    
    return lookup


def compute_coordinate_stats(lookup):
    """Compute normalization statistics from actual coordinate data."""
    if not lookup:
        return {"lat_mean": 0.0, "lat_std": 15.0, "lon_mean": 180.0, "lon_std": 100.0}
    
    lats = [v["lat"] for v in lookup.values()]
    lons = [v["lon"] for v in lookup.values()]
    
    import numpy as np
    lat_mean = float(np.mean(lats))
    lat_std = float(np.std(lats))
    lon_mean = float(np.mean(lons))
    lon_std = float(np.std(lons))
    
    # Guard against zero std
    lat_std = max(lat_std, 1.0)
    lon_std = max(lon_std, 1.0)
    
    print(f"\n  Coordinate Statistics:")
    print(f"    Lat: mean={lat_mean:.2f}°, std={lat_std:.2f}°, range=[{min(lats):.1f}, {max(lats):.1f}]")
    print(f"    Lon: mean={lon_mean:.2f}°, std={lon_std:.2f}°, range=[{min(lons):.1f}, {max(lons):.1f}]")
    
    return {
        "lat_mean": round(lat_mean, 4),
        "lat_std": round(lat_std, 4),
        "lon_mean": round(lon_mean, 4),
        "lon_std": round(lon_std, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build per-timestep coordinate lookup from IBTrACS')
    parser.add_argument('--root_dir', type=str,
                        default=r'/kaggle/input/datasets/meheruzannat/cyclone-dataset',
                        help='Root directory of the cyclone dataset')
    parser.add_argument('--years', nargs='+',
                        default=['2006', '2007', '2008', '2009', '2010',
                                 '2011', '2012'],
                        help='Year folders to scan')
    parser.add_argument('--output', type=str, default='coordinate_lookup.json',
                        help='Output JSON path')
    parser.add_argument('--ibtracs_cache', type=str,
                        default='ibtracs_all.csv',
                        help='Local path to cache the IBTrACS CSV')
    parser.add_argument('--max_gap_hours', type=int, default=3,
                        help='Maximum hours gap for timestamp matching')
    args, _ = parser.parse_known_args()  # parse_known_args to work in Colab/Jupyter

    print(f"{'='*60}")
    print("Building Coordinate Lookup from IBTrACS")
    print(f"{'='*60}")
    print(f"  Dataset root: {args.root_dir}")
    print(f"  Years: {args.years}")
    print(f"  Output: {args.output}")
    print()

    # Step 1: Download IBTrACS
    print("[1/4] Downloading IBTrACS data...")
    ibtracs_path = download_ibtracs(args.ibtracs_cache)

    # Step 2: Parse IBTrACS
    print("\n[2/4] Parsing IBTrACS CSV...")
    storm_tracks = parse_ibtracs_csv(ibtracs_path)

    # Step 3: Scan dataset
    print("\n[3/4] Scanning dataset directory...")
    dataset_entries = scan_dataset_timestamps(args.root_dir, args.years)

    # Step 4: Match and build lookup
    print("\n[4/4] Matching dataset to IBTrACS tracks...")
    lookup = build_lookup(storm_tracks, dataset_entries, args.max_gap_hours)

    # Compute normalization stats from actual data
    coord_stats = compute_coordinate_stats(lookup)

    # Save output
    output = {
        "metadata": {
            "ibtracs_version": "v04r01",
            "max_gap_hours": args.max_gap_hours,
            "num_entries": len(lookup),
            "years": args.years,
        },
        "coord_stats": coord_stats,
        "lookup": lookup,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Coordinate lookup saved to: {args.output}")
    print(f"  Entries: {len(lookup):,}")
    print(f"  Coord stats: {coord_stats}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
