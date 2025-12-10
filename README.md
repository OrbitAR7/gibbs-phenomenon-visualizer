# Gibbs Phenomenon Visualizer

A Python tool for visualizing the **Gibbs phenomenon** (ringing artifacts) in spherical harmonic reconstructions of geographical boundaries. This tool demonstrates how truncating spherical harmonic series at different degrees affects the reconstruction quality of country boundaries.

## Overview

The Gibbs phenomenon occurs when approximating a discontinuous function (like a country boundary mask) using a finite number of spherical harmonics. This creates characteristic "ringing" oscillations near sharp edges. This tool:

- Converts KML boundary files to XYZ format
- Creates high-resolution masks from boundary polygons
- Performs spherical harmonic analysis
- Visualizes ringing artifacts at different truncation degrees (Lmax = 30, 60, 120, 180)
- Generates both regional (zoomed) and global visualizations

## Example Output

The tool generates three types of visualizations:

1. **1D Line Plots** - Cross-sectional slices showing ringing at different degrees
2. **2D Regional View** - Zoomed comparison of original vs reconstructions
3. **2D Global View** - Full Earth visualization showing how ringing artifacts spread globally

## Installation


### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gibbs-phenomenon-visualizer.git
cd gibbs-phenomenon-visualizer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start with KML Files

The easiest way to analyze a country boundary:

```bash
python kml_analyzer.py "kml files/Saudi_Arabia.kml"
```

This will:
- Convert the KML file to XYZ format
- Create boundary mask
- Compute spherical harmonics
- Generate all visualizations in the `output/` folder

### Using Pre-existing XYZ Files

If you already have an XYZ boundary file:

```bash
python gibbs_visualizer.py boundary.xyz --country "Country Name"
```

### Command-Line Options

#### kml_analyzer.py
```bash
python kml_analyzer.py <kml_file> [OPTIONS]

Options:
  --output-dir DIR       Output directory (default: output)
  --save-data           Save analysis data (.npy, .json)
  --resolution FLOAT    Grid resolution in degrees (default: 0.25)
```

#### gibbs_visualizer.py
```bash
python gibbs_visualizer.py <xyz_file> [OPTIONS]

Options:
  --country NAME        Country/Region name (default: Region)
  --output-dir DIR      Output directory (default: output)
  --save-data          Save analysis data (.npy, .json)
  --resolution FLOAT    Grid resolution in degrees (default: 0.25)
```

## 📁 Examples

Analyze multiple countries:

```bash
# Saudi Arabia
python kml_analyzer.py "kml files/Saudi_Arabia.kml"

# India
python kml_analyzer.py "kml files/India.kml"
```

Save analysis data for later use:

```bash
python kml_analyzer.py "kml files/Saudi_Arabia.kml" --save-data
```

Use custom resolution (higher = more detail, slower):

```bash
python kml_analyzer.py "kml files/Saudi_Arabia.kml" --resolution 0.1
```

## Scientific Background

### What is the Gibbs Phenomenon?

The Gibbs phenomenon is an overshoot (ringing) that occurs when approximating a discontinuous function with a Fourier series or spherical harmonics. Key characteristics:

- **Overshoot**: ~9% of the jump discontinuity
- **Frequency**: Oscillations increase with higher truncation degree
- **Persistence**: Does not disappear as degree increases, only becomes more concentrated near the discontinuity

### Spherical Harmonics

Spherical harmonics are the basis functions for representing functions on a sphere, analogous to Fourier series for periodic functions. They are widely used in:

- Geophysics (gravity and magnetic field modeling)
- Climate science (global temperature/pressure fields)
- Geodesy (Earth's shape and gravity)
- Computer graphics (lighting and environment mapping)

### Truncation Effects

This tool demonstrates how different truncation degrees (Lmax) affect reconstruction:

- **Lmax = 30**: Strong ringing, visible global artifacts
- **Lmax = 60**: Moderate ringing, oscillations still prominent
- **Lmax = 120**: Reduced ringing, better boundary definition
- **Lmax = 180**: Minimal ringing, accurate reconstruction

## Technical Details

### Algorithm

1. **Boundary Loading**: Parse KML files to extract polygon coordinates
2. **Mask Creation**: Generate binary mask on regular lat/lon grid (1 inside, 0 outside)
3. **Global Mapping**: Map regional mask to global grid (required for spherical harmonics)
4. **SH Expansion**: Compute spherical harmonic coefficients using `SHExpandDH` (sampling=2)
5. **SH Synthesis**: Reconstruct at different degrees using `MakeGridDH`
6. **Visualization**: Generate 1D slices and 2D maps (regional + global)

### Grid Resolution

- Default: 0.25° (~28 km at equator)
- Adjustable via `--resolution` parameter
- Higher resolution = more accurate but slower

### Performance Optimizations

- Vectorized point-in-polygon testing using `shapely.prepared`
- Efficient grid mapping with NumPy operations
- Parallel-ready design (can be extended)

## Output Files

### Visualization Files
- `gibbs_comparison_1d_<country>.png` - Line plots at 4 degrees
- `gibbs_comparison_2d_<country>.png` - Regional 2D comparison
- `gibbs_comparison_2d_global_<country>.png` - Global 2D comparison

### Data Files (with --save-data)
- `<country>_mask.npy` - Regional boundary mask
- `<country>_clm_coefficients.npy` - Spherical harmonic coefficients
- `<country>_metadata.json` - Analysis metadata

### Output Example
<img width="2539" height="751" alt="gibbs_comparison_2d_saudi_arabia" src="https://github.com/user-attachments/assets/3c022053-c1c3-41b7-98d5-3fa0cdbf03a1" />
<img width="2985" height="865" alt="gibbs_comparison_2d_global_saudi_arabia" src="https://github.com/user-attachments/assets/c8e162c1-8d77-438b-ac38-ac754f363335" />
<img width="2386" height="1507" alt="gibbs_comparison_1d_saudi_arabia" src="https://github.com/user-attachments/assets/94516661-ea5c-4eca-8b26-7c3cdb0ca062" />


## Acknowledgments

- Built using [SHTOOLS](https://shtools.github.io/SHTOOLS/) for spherical harmonic transformations
- Cartographic visualizations powered by [Cartopy](https://scitools.org.uk/cartopy/)

