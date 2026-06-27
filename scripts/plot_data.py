import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# --- Configuration ---
ROOT_DIR = r"E:\setcd_locked"
YEARS = ['2022_0']

# Sample paths
era5_path = r"E:\setcd_locked\2022_0\2021323S10103\2021-11-19 06_00_00\ERA5_data.npy"
gridsat_path = r"E:\setcd_locked\2022_0\2021323S10103\2021-11-19 06_00_00\GRIDSAT_data.npy"

# --- Collect dataset statistics ---
def collect_dataset_statistics(root_dir, years):
    stats = {
        'total_storms': set(),
        'total_timesteps': 0,
        'storms_per_year': {},
        'timesteps_per_storm': {},
        'time_intervals': []
    }
    
    root_path = Path(root_dir)
    
    for year_folder in years:
        year_path = root_path / year_folder
        if not year_path.exists():
            continue
        
        nested_path = year_path / year_folder
        if nested_path.exists():
            year_path = nested_path
        
        storm_count = 0
        for cyclone_folder in year_path.iterdir():
            if not cyclone_folder.is_dir():
                continue
            
            storm_name = cyclone_folder.name
            stats['total_storms'].add(storm_name)
            storm_count += 1
            
            timestep_folders = sorted([f for f in cyclone_folder.iterdir() if f.is_dir()])
            timestep_count = len(timestep_folders)
            stats['total_timesteps'] += timestep_count
            stats['timesteps_per_storm'][storm_name] = timestep_count
            
            if len(timestep_folders) >= 2:
                try:
                    time1 = datetime.strptime(timestep_folders[0].name, '%Y-%m-%d %H_%M_%S')
                    time2 = datetime.strptime(timestep_folders[1].name, '%Y-%m-%d %H_%M_%S')
                    interval_hours = (time2 - time1).total_seconds() / 3600
                    stats['time_intervals'].append(interval_hours)
                except:
                    pass
        
        stats['storms_per_year'][year_folder] = storm_count
    
    stats['total_storms'] = len(stats['total_storms'])
    stats['avg_interval_hours'] = np.mean(stats['time_intervals']) if stats['time_intervals'] else 0
    
    return stats

# Collect statistics
stats = collect_dataset_statistics(ROOT_DIR, YEARS)

# Load data
era5 = np.load(era5_path)
gridsat = np.load(gridsat_path)

# Extract variables
u10 = era5[0]
v10 = era5[1]
t2m = era5[2]
msl = era5[3]

ir = gridsat[0]
wv = gridsat[1]
vis = gridsat[2]

wind_speed = np.sqrt(u10**2 + v10**2)

# ============================================================
# FIGURE 1: DATA VISUALIZATIONS
# ============================================================
fig1 = plt.figure(figsize=(14, 7))
fig1.suptitle('Tropical Cyclone Data Visualization', fontsize=16, fontweight='bold', y=0.98)

# Create 2x4 grid
gs1 = fig1.add_gridspec(2, 4, hspace=0.25, wspace=0.25, top=0.93, bottom=0.08, left=0.05, right=0.98)

# Row 1: ERA5 Wind and Temperature
ax1 = fig1.add_subplot(gs1[0, 0])
im1 = ax1.imshow(u10, cmap='coolwarm', vmin=-20, vmax=20)
ax1.set_title('Wind U-Component\n(West ← → East)', fontsize=12, fontweight='bold')
ax1.axis('off')
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('m/s', fontsize=10)

ax2 = fig1.add_subplot(gs1[0, 1])
im2 = ax2.imshow(v10, cmap='coolwarm', vmin=-20, vmax=20)
ax2.set_title('Wind V-Component\n(South ← → North)', fontsize=12, fontweight='bold')
ax2.axis('off')
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('m/s', fontsize=10)

ax3 = fig1.add_subplot(gs1[0, 2])
im3 = ax3.imshow(wind_speed, cmap='YlOrRd')
ax3.set_title('Total Wind Speed', fontsize=12, fontweight='bold')
ax3.axis('off')
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('m/s', fontsize=10)

ax4 = fig1.add_subplot(gs1[0, 3])
im4 = ax4.imshow(t2m, cmap='RdYlBu_r')
temp_celsius = np.mean(t2m) - 273.15
ax4.set_title(f'Surface Temperature (2m)\nAvg: {temp_celsius:.1f}°C', fontsize=12, fontweight='bold')
ax4.axis('off')
cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
cbar4.set_label('Kelvin', fontsize=10)

# Row 2: Pressure and GRIDSAT
ax5 = fig1.add_subplot(gs1[1, 0])
im5 = ax5.imshow(msl, cmap='viridis')
ax5.set_title('Sea Level Pressure', fontsize=12, fontweight='bold')
ax5.axis('off')
cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
cbar5.set_label('Pa', fontsize=10)

ax6 = fig1.add_subplot(gs1[1, 1])
im6 = ax6.imshow(ir, cmap='gray_r')
ax6.set_title('GRIDSAT: Infrared\n(Cloud Top Temperature)', fontsize=12, fontweight='bold')
ax6.axis('off')
cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
cbar6.set_label('Brightness', fontsize=10)

ax7 = fig1.add_subplot(gs1[1, 2])
im7 = ax7.imshow(wv, cmap='BuPu')
ax7.set_title('GRIDSAT: Water Vapor\n(Moisture Content)', fontsize=12, fontweight='bold')
ax7.axis('off')
cbar7 = plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
cbar7.set_label('Intensity', fontsize=10)

ax8 = fig1.add_subplot(gs1[1, 3])
im8 = ax8.imshow(vis, cmap='gray')
ax8.set_title('GRIDSAT: Visible\n(Cloud Reflectance)', fontsize=12, fontweight='bold')
ax8.axis('off')
cbar8 = plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
cbar8.set_label('Reflectance', fontsize=10)

plt.savefig('cyclone_visualization_plots.png', dpi=150, bbox_inches='tight')
print("Saved: cyclone_visualization_plots.png")

# ============================================================
# FIGURE 2: DATASET INFORMATION
# ============================================================
fig2 = plt.figure(figsize=(16, 10))
fig2.suptitle('Dataset Statistics & Technical Information', fontsize=16, fontweight='bold', y=0.96)

gs2 = fig2.add_gridspec(4, 2, hspace=0.35, wspace=0.25, top=0.90, bottom=0.05, left=0.06, right=0.94)

# Panel 1: Dataset Overview
ax_stats = fig2.add_subplot(gs2[0, :])
ax_stats.axis('off')
stats_text = f"""DATASET OVERVIEW
{'-'*90}

Total Storms:           {stats['total_storms']}
Total Timesteps:        {stats['total_timesteps']:,}
Temporal Resolution:    {stats['avg_interval_hours']:.0f}-hour intervals (3-hourly recordings)
Spatial Resolution:     0.25 deg x 0.25 deg (~27.8 km at equator)
Grid Size:              {era5.shape[1]} x {era5.shape[2]} pixels
"""
ax_stats.text(0.02, 0.90, stats_text, transform=ax_stats.transAxes, fontsize=11,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4, pad=0.8))

# Panel 2: ERA5 Variables
ax_era5 = fig2.add_subplot(gs2[1:3, 0])
ax_era5.axis('off')
era5_text = """ERA5 REANALYSIS VARIABLES
{'-'*40}

Surface Level (10m/2m):

  - u10:  U-wind component (m/s)
          Horizontal wind (East-West)

  - v10:  V-wind component (m/s)
          Vertical wind (North-South)

  - t2m:  2-meter temperature (K)
          Surface air temperature

  - msl:  Mean sea level pressure (Pa)
          Atmospheric pressure


Total Variables: 69 channels
Data Shape: (69, H, W)
"""
ax_era5.text(0.02, 0.95, era5_text, transform=ax_era5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.4, pad=0.8))

# Panel 3: GRIDSAT Channels
ax_gridsat = fig2.add_subplot(gs2[1:3, 1])
ax_gridsat.axis('off')
gridsat_text = """GRIDSAT SATELLITE IMAGERY
{'-'*40}

Spectral Channels:

  - IR:   Infrared (~11 um)
          Cloud top temperature
          Bright = Cold/High clouds
          Dark = Warm/Low clouds

  - WV:   Water Vapor (~6.7 um)
          Atmospheric moisture
          Shows moisture distribution

  - VIS:  Visible (~0.6 um)
          Cloud reflectance
          Bright = Dense clouds


Total Channels: 3
Data Shape: (3, H, W)
"""
ax_gridsat.text(0.02, 0.95, gridsat_text, transform=ax_gridsat.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4, pad=0.8))





plt.show()

print("\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print("Two figures created:")
print("  1. cyclone_visualization_plots.png - Data visualizations")
print("  2. cyclone_dataset_info.png - Dataset statistics")
print("="*60)