# gibbs-phenomenon-visualizer

Visualizes **Gibbs ringing** when a sharp-edged regional mask, here a country boundary, is expanded in spherical harmonics and truncated at different maximum degrees.

This matters in satellite geodesy. GRACE/GRACE-FO gravity solutions are band-limited spherical-harmonic fields, so estimating mass change over a basin or country means multiplying a smooth field by a sharp mask. Truncation then produces ringing and **leakage** of signal across the boundary. This tool makes that effect visible for any boundary in a KML file.

<img width="2539" height="751" alt="Saudi Arabia mask and its spherical-harmonic reconstructions" src="https://github.com/user-attachments/assets/3c022053-c1c3-41b7-98d5-3fa0cdbf03a1" />
<img width="2985" height="865" alt="Global view of ringing for the Saudi Arabia mask" src="https://github.com/user-attachments/assets/c8e162c1-8d77-438b-ac38-ac754f363335" />
<img width="2386" height="1507" alt="Profiles across the boundary at several truncation degrees" src="https://github.com/user-attachments/assets/94516661-ea5c-4eca-8b26-7c3cdb0ca062" />

## How it works

1. **Boundary → mask.** Parse the KML polygon and rasterize it into a 1/0 mask on a regional grid (vectorized point-in-polygon with `shapely`).
2. **Regional → global grid.** Place the mask on a global 0.5° Driscoll–Healy grid (360 × 720).
3. **Analysis.** Expand with `pyshtools.expand.SHExpandDH`. This grid supports degrees up to 179.
4. **Synthesis.** Reconstruct with coefficients truncated at L<sub>max</sub> = 30, 60, 120 and the full expansion.
5. **Plots.** Profiles across the boundary, regional 2-D maps, and a global map showing how far the ringing spreads.

## What to look for

- The overshoot at the edge stays at roughly 9% of the jump however high L<sub>max</sub> goes. Raising the degree narrows the ringing toward the boundary but does not remove it.
- At low degree (L<sub>max</sub> = 30, roughly 670 km half-wavelength), oscillations extend far outside the region, which is the leakage problem in basin averages.
- Smoothing filters (e.g., Gaussian) trade ringing for blur. That trade-off is the next thing to add (see below).

## Quick start

```bash
git clone https://github.com/OrbitAR7/gibbs-phenomenon-visualizer.git
cd gibbs-phenomenon-visualizer
pip install -r requirements.txt        # numpy, matplotlib, cartopy, pyshtools, shapely, pandas

python kml_analyzer.py data/Saudi_Arabia.kml
python kml_analyzer.py data/India.kml --save-data
```

Figures are written to `output/`. `gibbs_visualizer.py` does the same starting from an XYZ boundary file (`python gibbs_visualizer.py boundary.xyz --country "Name"`).

| Option | Default | Meaning |
|---|---|---|
| `--resolution` | 0.25 | Regional mask grid spacing in degrees. Smaller values give finer masks but run slower |
| `--output-dir` | `output` | Where figures go |
| `--save-data` | off | Also save the mask, SH coefficients (`.npy`) and metadata (`.json`) |

## Limitations and next steps

- The spherical-harmonic step always runs on the 0.5° global grid, so `--resolution` refines only the regional mask that is mapped onto it.
- The mask is a pure indicator (1 inside, 0 outside). It doesn't yet weight by area or apply ocean/land separation.
- Planned: Gaussian and other smoothing filters, a leakage estimate (the fraction of the mask's energy outside the boundary vs. L<sub>max</sub>), and a comparison against a synthetic GRACE-like mass-change signal.

## Built with

[SHTOOLS / pyshtools](https://shtools.github.io/SHTOOLS/) · [Cartopy](https://scitools.org.uk/cartopy/) · [Shapely](https://shapely.readthedocs.io/)

## License

MIT. See `LICENSE`.
