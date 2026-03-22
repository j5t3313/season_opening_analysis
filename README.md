# F1 2026 Season Opening Analysis
 
Analysis code for the Substack article examining the first two races of the 2026 F1 season (Australia and China). Covers reliability, pace consistency, and energy deployment behavior across the grid.
 
## Data Sources
 
- **FastF1 (v3.8.1):** Qualifying telemetry and race lap data for the 2026 Australian and Chinese Grands Prix
- **FIA Final Race Classification documents:** Official results including DNF lap, DNS status, and finishing classification for both races
 
## Scripts
 
### reliability.py
 
Kaplan-Meier survival analysis on race reliability. Each car-start is an observation. DNFs are events; classified finishers are right-censored at race completion. DNS entries are excluded from the survival model and reported separately.
 
The time axis is normalized to percentage of race completed (laps / total race laps) so the two races are comparable. All classification data is embedded directly in the script from the official FIA documents.
 
Outputs:
- Kaplan-Meier curves grouped by PU manufacturer
- Kaplan-Meier curves grouped by works vs. customer status
- Pairwise log-rank test results
- DNS and DNF summaries to console
 
### pace_variance.py
 
Computes the coefficient of variation (CV = std/mean) of sector times and lap times for each driver across clean race laps. Lower CV indicates more consistent pace.
 
Lap filtering:
- Only laps where FastF1's `IsAccurate` flag is True
- Pit in/out laps excluded
- Minimum 5 clean laps required per driver
- Drivers with fewer than 15 clean laps are flagged in all visuals as unreliable estimates
 
Outputs:
- Horizontal bar chart ranking drivers by lap time CV per race (low-lap-count bars hatched)
- Heatmap of sector-level CV by driver per race
- Dot plot of CV grouped by PU manufacturer across both races (low-lap-count entries marked with x)
 
### deployment_gap.py
 
Compares super-clipping behavior across the grid using qualifying telemetry. For each driver's fastest qualifying lap (filtered to within 107% of the session fastest, matching the FIA qualifying cutoff), speed traces are extracted on designated straights.
 
Super-clipping is detected where throttle >= 95% and dSpeed/dDistance < -0.02. The speed signal is smoothed with a Savitzky-Golay filter (window 13, polynomial order 2) before gradient computation. Straight-level observations with less than 30 km/h of speed loss are excluded as non-representative.
 
Straight definitions:
 
| Circuit | Straight | Distance range (m) |
|---|---|---|
| Australia | T8-T9 back straight | 3400-4200 |
| China | T13-T14 back straight | 3800-4700 |
| China | Pit straight | 0-600 |
 
Outputs:
- Speed trace overlays per straight, color-coded by team (solid/dashed for driver 1/driver 2)
- Summary dot plot of clipping severity and frequency grouped by PU manufacturer
 
## PU Manufacturer Mapping
 
| PU | Teams |
|---|---|
| Mercedes | Mercedes, McLaren, Williams, Alpine |
| Ferrari | Ferrari, Haas, Cadillac |
| Ford x RBPT | Red Bull, Racing Bulls |
| Honda | Aston Martin |
| Audi | Audi |
 
## Team Colors (speed trace overlays)
 
| Team | Hex |
|---|---|
| Mercedes | #27F4D2 |
| Ferrari | #E8002D |
| Red Bull | #0f1b3c |
| McLaren | #FF8000 |
| Alpine | #FF87BC |
| Aston Martin | #006F62 |
| Haas | #B6BABD |
| Racing Bulls | #6692FF |
| Williams | #1868DB |
| Audi | #00E701 |
| Cadillac | #C5A647 |
 
## Limitations
 
- Two races is a minimal sample. All findings are preliminary.
- FastF1 telemetry does not include battery state-of-charge or MGU-K mode. Energy state classification is inferred from kinematics.
- DNF cause is not available from the FIA classification documents. PU attribution is assumed by team.
- Pace variance is confounded by tire compound, tire age, fuel load, traffic, and safety car timing.
- Straight segment boundaries are manually defined.
- Survival analysis sample size (n=38 car-starts after DNS exclusion) limits statistical power.
 
## Setup
 
```
pip install -r substack_requirements.txt
```
 
`deployment_gap.py` and `pace_variance.py` require FastF1 3.8.1+ for 2026 season telemetry support. `reliability.py` has no external data dependencies.
 
## Usage
 
```
python reliability.py
python pace_variance.py
python deployment_gap.py
```
 
Each FastF1 script creates a `cache/` directory on first run. All plots are saved as PNG files in the working directory.