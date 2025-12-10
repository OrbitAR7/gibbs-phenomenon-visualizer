#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gibbs Phenomenon Analyzer with KML to XYZ Conversion
Converts KML boundary files to XYZ format and runs spherical harmonic analysis

Usage:
    python kml_analyzer.py <kml_file> [--output-dir OUTPUT_DIR]
    
Example:
    python kml_analyzer.py "kml files/Saudi_Arabia.kml"
    python kml_analyzer.py "kml files/Italy.kml" --output-dir output/
"""

import argparse
import sys
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyshtools
from shapely.geometry import Polygon, Point
from shapely.prepared import prep


def parse_kml(kml_file):
    """
    Parse KML file and extract coordinates from polygons
    
    Parameters:
    -----------
    kml_file : str
        Path to KML file
        
    Returns:
    --------
    coordinates : list of tuples
        List of (longitude, latitude) tuples
    """
    
    tree = ET.parse(kml_file)
    root = tree.getroot()
    
    # Define KML namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    coordinates = []
    
    # Look for LinearRing coordinates (polygons)
    for linear_ring in root.findall('.//kml:LinearRing/kml:coordinates', ns):
        if linear_ring.text:
            coord_strings = linear_ring.text.strip().split()
            for coord_str in coord_strings:
                parts = coord_str.split(',')
                lon, lat = float(parts[0]), float(parts[1])
                coordinates.append((lon, lat))
            if coordinates:
                break
    
    # If no LinearRing found, try LineString
    if not coordinates:
        for line_string in root.findall('.//kml:LineString/kml:coordinates', ns):
            if line_string.text:
                coord_strings = line_string.text.strip().split()
                for coord_str in coord_strings:
                    parts = coord_str.split(',')
                    lon, lat = float(parts[0]), float(parts[1])
                    coordinates.append((lon, lat))
                if coordinates:
                    break
    
    if not coordinates:
        raise ValueError(f"No coordinates found in {kml_file}")
    
    return coordinates


def save_xyz(coordinates, output_file, value=0):
    """
    Save coordinates to XYZ format
    
    Parameters:
    -----------
    coordinates : list of tuples
        List of (longitude, latitude) tuples
    output_file : str
        Output XYZ file path
    value : float
        Value for the third column (default 0)
    """
    
    with open(output_file, 'w') as f:
        for lon, lat in coordinates:
            f.write(f"{lon} {lat} {value}\n")
    
    print(f"✓ Saved {len(coordinates)} points to {output_file}")


def load_boundary_xyz(xyz_file):
    """
    Load boundary coordinates from XYZ file
    
    Parameters:
    -----------
    xyz_file : str
        Path to XYZ file
        
    Returns:
    --------
    boundary_data : pd.DataFrame
        DataFrame with Longitude, Latitude, Value columns
    """
    boundary_data = pd.read_csv(xyz_file, header=None, sep='\s+',
                                names=['Longitude', 'Latitude', 'Value'])
    return boundary_data


def create_mask(boundary_data, resolution=0.25):
    """
    Create a high-resolution gridded mask from boundary polygon
    
    Parameters:
    -----------
    boundary_data : pd.DataFrame
        Boundary coordinates
    resolution : float
        Grid resolution in degrees (default 0.25)
        
    Returns:
    --------
    mask : np.ndarray
        2D mask array (1 inside, 0 outside)
    lat_grid : np.ndarray
        Latitude coordinates
    lon_grid : np.ndarray
        Longitude coordinates
    """
    
    # Create polygon
    polygon = Polygon(zip(boundary_data['Longitude'], boundary_data['Latitude']))
    
    # Validate polygon
    if not polygon.is_valid:
        print("Warning: Polygon is invalid, attempting to fix...")
        polygon = polygon.buffer(0)
    
    # Create regional grid
    lon_min, lat_min, lon_max, lat_max = polygon.bounds
    lon_min -= 2
    lon_max += 2
    lat_min -= 2
    lat_max += 2
    
    nlat_grid = int((lat_max - lat_min) / resolution) + 1
    nlon_grid = int((lon_max - lon_min) / resolution) + 1
    
    lat_grid = np.linspace(lat_max, lat_min, nlat_grid)
    lon_grid = np.linspace(lon_min, lon_max, nlon_grid)
    
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Create mask using prepared geometry (fast)
    print(f"Creating mask: {nlat_grid}×{nlon_grid} = {nlat_grid*nlon_grid:,} points...")
    prepared_polygon = prep(polygon)
    
    lon_flat = lon_mesh.ravel()
    lat_flat = lat_mesh.ravel()
    mask_flat = np.array([prepared_polygon.contains(Point(lon, lat)) 
                          for lon, lat in zip(lon_flat, lat_flat)], dtype=float)
    mask = mask_flat.reshape((nlat_grid, nlon_grid))
    
    print(f"✓ Mask created: {mask.sum():.0f} grid points inside boundary")
    print(f"Polygon bounds (lon, lat): ({lon_min:.2f}, {lat_min:.2f}) to ({lon_max:.2f}, {lat_max:.2f})")
    
    return mask, lat_grid, lon_grid


def create_global_mask(mask, lat_grid, lon_grid, nlat_global=360, nlon_global=720):
    """
    Map regional mask to global grid for SHExpandDH compatibility
    
    Parameters:
    -----------
    mask : np.ndarray
        Regional mask
    lat_grid : np.ndarray
        Regional latitude grid
    lon_grid : np.ndarray
        Regional longitude grid
    nlat_global : int
        Global latitude resolution
    nlon_global : int
        Global longitude resolution
        
    Returns:
    --------
    mask_global : np.ndarray
        Global mask array
    """
    
    lat_global = np.linspace(90, -90, nlat_global)
    lon_global = np.linspace(0, 360, nlon_global, endpoint=False)
    
    mask_global = np.zeros((nlat_global, nlon_global))
    
    for i, lat_r in enumerate(lat_grid):
        for j, lon_r in enumerate(lon_grid):
            lat_idx = np.argmin(np.abs(lat_global - lat_r))
            lon_idx = np.argmin(np.abs(lon_global - lon_r))
            mask_global[lat_idx, lon_idx] = mask[i, j]
    
    return mask_global


def compute_spherical_harmonics(mask_global):
    """
    Compute spherical harmonic coefficients
    
    Parameters:
    -----------
    mask_global : np.ndarray
        Global mask array
        
    Returns:
    --------
    clm_full : np.ndarray
        Spherical harmonic coefficients
    """
    
    print("Computing spherical harmonics...")
    clm_full = pyshtools.expand.SHExpandDH(mask_global, sampling=2)
    print("✓ Spherical harmonic coefficients computed")
    
    return clm_full


def create_line_comparison_plot(mask, lat_grid, lon_grid, clm_full, country_name, output_dir):
    """
    Create 1D line plot comparing ringing at different degrees
    """
    
    desired_latitude = (lat_grid.max() + lat_grid.min()) / 2  # Central latitude
    
    lat_idx_orig = np.argmin(np.abs(lat_grid - desired_latitude))
    slice_original = mask[lat_idx_orig, :]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    degrees = [30, 60, 120, 180]
    axes_flat = axes.flatten()
    
    for idx, lmax in enumerate(degrees):
        ax = axes_flat[idx]
        
        # Truncate and synthesize
        clm_truncated = clm_full[:, :lmax+1, :lmax+1]
        grid_synth = pyshtools.expand.MakeGridDH(clm_truncated, lmax=lmax, sampling=2)
        
        nlat_synth, nlon_synth = grid_synth.shape
        lats_synth = np.linspace(90, -90, nlat_synth)
        lat_idx_synth = np.argmin(np.abs(lats_synth - desired_latitude))
        slice_synth = grid_synth[lat_idx_synth, :]
        
        # Plot
        ax.plot(lon_grid, slice_original, 'k-', linewidth=2.5, label='Original Boundary')
        ax.plot(np.linspace(0, 360, nlon_synth), slice_synth, 'r-', linewidth=2,
                label=f'SH Reconstruction (Lmax={lmax})')
        
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Grid Value', fontsize=11)
        ax.set_title(f'Ringing at Lmax={lmax}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Zoom to region of interest
        lon_min = lon_grid.min()
        lon_max = lon_grid.max()
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(-0.5, 1.5)
    
    plt.suptitle(f'{country_name}: Gibbs Phenomenon at {desired_latitude:.1f}°',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_file = Path(output_dir) / f'gibbs_comparison_1d_{country_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_2d_comparison_plot(mask, lat_grid, lon_grid, clm_full, country_name, output_dir):
    """
    Create 2D grid visualization comparing original vs reconstructions
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    lon_min, lon_max = lon_grid.min(), lon_grid.max()
    lat_min, lat_max = lat_grid.min(), lat_grid.max()
    
    # Create meshgrid for regional data
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Original mask
    ax = axes[0]
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    im = ax.pcolormesh(lon_mesh, lat_mesh, mask, cmap='gray_r', transform=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_title('Original Boundary', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, orientation='horizontal', label='Mask Value', shrink=0.8)
    
    # Lmax=30 (high ringing)
    lmax = 30
    clm_truncated = clm_full[:, :lmax+1, :lmax+1]
    grid_synth_30 = pyshtools.expand.MakeGridDH(clm_truncated, lmax=lmax, sampling=2)
    
    ax = axes[1]
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    lats_synth = np.linspace(90, -90, grid_synth_30.shape[0])
    lons_synth = np.linspace(0, 360, grid_synth_30.shape[1])
    lons_mesh_synth, lats_mesh_synth = np.meshgrid(lons_synth, lats_synth)
    im = ax.pcolormesh(lons_mesh_synth, lats_mesh_synth, grid_synth_30, cmap='gray_r',
                       vmin=-0.5, vmax=1.5, transform=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_title(f'SH Reconstruction (Lmax=30)\nStrong Ringing', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, orientation='horizontal', label='Value', shrink=0.8)
    
    # Lmax=180 (minimal ringing)
    lmax = 180
    clm_truncated = clm_full[:, :lmax+1, :lmax+1]
    grid_synth_180 = pyshtools.expand.MakeGridDH(clm_truncated, lmax=lmax, sampling=2)
    
    ax = axes[2]
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    lats_synth = np.linspace(90, -90, grid_synth_180.shape[0])
    lons_synth = np.linspace(0, 360, grid_synth_180.shape[1])
    lons_mesh_synth, lats_mesh_synth = np.meshgrid(lons_synth, lats_synth)
    im = ax.pcolormesh(lons_mesh_synth, lats_mesh_synth, grid_synth_180, cmap='gray_r',
                       vmin=-0.5, vmax=1.5, transform=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_title(f'SH Reconstruction (Lmax=180)\nMinimal Ringing', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, orientation='horizontal', label='Value', shrink=0.8)
    
    plt.suptitle(f'{country_name}: Gibbs Phenomenon Decreases with Higher Degree',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path(output_dir) / f'gibbs_comparison_2d_{country_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_2d_global_comparison_plot(mask_global, clm_full, country_name, output_dir):
    """
    Create GLOBAL 2D grid visualization showing ringing artifacts across the entire planet
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.Robinson()})
    
    # Original global mask
    ax = axes[0]
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
    lats_global = np.linspace(90, -90, mask_global.shape[0])
    lons_global = np.linspace(0, 360, mask_global.shape[1], endpoint=False)
    lons_mesh, lats_mesh = np.meshgrid(lons_global, lats_global)
    im = ax.pcolormesh(lons_mesh, lats_mesh, mask_global, cmap='tab20c', 
                       vmin=-1, vmax=1, transform=ccrs.PlateCarree())
    ax.set_title('Original Mask (Global)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, orientation='horizontal', label='Mask Value', shrink=0.6)
    
    # Lmax=30 GLOBAL - shows strong ringing artifacts
    lmax = 30
    clm_truncated = clm_full[:, :lmax+1, :lmax+1]
    grid_synth_30 = pyshtools.expand.MakeGridDH(clm_truncated, lmax=lmax, sampling=2)
    
    ax = axes[1]
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
    lats_synth = np.linspace(90, -90, grid_synth_30.shape[0])
    lons_synth = np.linspace(0, 360, grid_synth_30.shape[1], endpoint=False)
    lons_mesh, lats_mesh = np.meshgrid(lons_synth, lats_synth)
    im = ax.pcolormesh(lons_mesh, lats_mesh, grid_synth_30, cmap='tab20c', 
                       vmin=-1, vmax=1, transform=ccrs.PlateCarree())
    ax.set_title(f'SH Reconstruction (Lmax=30)\nStrong Ringing - Global Artifacts', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, orientation='horizontal', label='Value', shrink=0.6)
    
    # Lmax=180 GLOBAL - minimal ringing
    lmax = 180
    clm_truncated = clm_full[:, :lmax+1, :lmax+1]
    grid_synth_180 = pyshtools.expand.MakeGridDH(clm_truncated, lmax=lmax, sampling=2)
    
    ax = axes[2]
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
    lats_synth = np.linspace(90, -90, grid_synth_180.shape[0])
    lons_synth = np.linspace(0, 360, grid_synth_180.shape[1], endpoint=False)
    lons_mesh, lats_mesh = np.meshgrid(lons_synth, lats_synth)
    im = ax.pcolormesh(lons_mesh, lats_mesh, grid_synth_180, cmap='tab20c', 
                       vmin=-1, vmax=1, transform=ccrs.PlateCarree())
    ax.set_title(f'SH Reconstruction (Lmax=180)\nMinimal Ringing - Accurate', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, orientation='horizontal', label='Value', shrink=0.6)
    
    plt.suptitle(f'{country_name}: Gibbs Phenomenon in Global View\nNotice ringing artifacts spread across entire planet at low degree', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = Path(output_dir) / f'gibbs_comparison_2d_global_{country_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def save_data(mask, lat_grid, lon_grid, clm_full, country_name, output_dir):
    """
    Save analysis data to files
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = country_name.lower().replace(" ", "_")
    
    # Save arrays
    np.save(output_dir / f'{prefix}_mask.npy', mask)
    np.save(output_dir / f'{prefix}_clm_coefficients.npy', clm_full)
    
    # Save metadata
    metadata = {
        'country': country_name,
        'nlat_regional': len(lat_grid),
        'nlon_regional': len(lon_grid),
        'lat_range': [float(lat_grid.min()), float(lat_grid.max())],
        'lon_range': [float(lon_grid.min()), float(lon_grid.max())],
        'mask_points_inside': int(mask.sum()),
        'grid_resolution': 0.25,
    }
    
    with open(output_dir / f'{prefix}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved data files:")
    print(f"  - {prefix}_mask.npy")
    print(f"  - {prefix}_clm_coefficients.npy")
    print(f"  - {prefix}_metadata.json")


def main():
    parser = argparse.ArgumentParser(
        description='Gibbs Phenomenon Analyzer - KML to XYZ to Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kml_analyzer.py "kml files/Saudi_Arabia.kml"
  python kml_analyzer.py "kml files/Italy.kml" --output-dir output/
        """)
    
    parser.add_argument('kml_file', help='KML boundary file')
    parser.add_argument('--output-dir', default='output', help='Output directory for plots and data (default: output)')
    parser.add_argument('--save-data', action='store_true', default=True, help='Save analysis data (default: True)')
    parser.add_argument('--resolution', type=float, default=0.25, help='Grid resolution in degrees (default: 0.25)')
    
    args = parser.parse_args()
    
    try:
        # Get country name from filename
        kml_path = Path(args.kml_file)
        country_name = kml_path.stem  # Remove .kml extension
        
        print(f"\n{'='*70}")
        print(f"Gibbs Phenomenon Analyzer: {country_name}")
        print(f"{'='*70}\n")
        
        # Step 1: Convert KML to XYZ
        print("Step 1: Converting KML to XYZ format...")
        print(f"Reading KML file: {args.kml_file}")
        coordinates = parse_kml(args.kml_file)
        print(f"✓ Extracted {len(coordinates)} boundary points\n")
        
        # Save XYZ temporarily
        xyz_file = Path('temp_boundary.xyz')
        save_xyz(coordinates, str(xyz_file))
        print()
        
        # Step 2: Load XYZ and create mask
        print("Step 2: Creating mask from boundary...")
        boundary_data = load_boundary_xyz(str(xyz_file))
        mask, lat_grid, lon_grid = create_mask(boundary_data, resolution=args.resolution)
        print()
        
        # Step 3: Create global mask
        print("Step 3: Mapping to global grid...")
        mask_global = create_global_mask(mask, lat_grid, lon_grid)
        print(f"✓ Global mask shape: {mask_global.shape}\n")
        
        # Step 4: Compute spherical harmonics
        print("Step 4: Computing spherical harmonics...")
        start = time.time()
        clm_full = compute_spherical_harmonics(mask_global)
        elapsed = time.time() - start
        print(f"✓ SH computation took {elapsed:.2f}s\n")
        
        # Step 5: Create visualizations
        print("Step 5: Creating visualizations...")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        create_line_comparison_plot(mask, lat_grid, lon_grid, clm_full, country_name, args.output_dir)
        create_2d_comparison_plot(mask, lat_grid, lon_grid, clm_full, country_name, args.output_dir)
        create_2d_global_comparison_plot(mask_global, clm_full, country_name, args.output_dir)
        print()
        
        # Step 6: Save data
        if args.save_data:
            print("Step 6: Saving analysis data...")
            save_data(mask, lat_grid, lon_grid, clm_full, country_name, f'{args.output_dir}/data')
            print()
        
        # Cleanup
        xyz_file.unlink()
        
        print(f"{'='*70}")
        print("✓ Analysis complete!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
