from rocketpy import Environment, SolidMotor, Rocket, Flight, GenericMotor
import matplotlib
from rocketpy import GenericMotor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# setting up environment for simulation (roughly corresponds to Midland, TX on a hot summer day)

env = Environment(latitude= 32.000507, longitude=-102.077408, elevation=848)
env.set_date((2027, 6, 12, 12), timezone="America/Chicago")

# Temperature profile in Kelvin vs. height above sea level (m).
# 30 °C at the ground (848 m), cooling ~6.5 °C per km up to ~11 km.


ground_T = 30 + 273.15
T_sea = ground_T + 0.0065 * 848
temperature_profile = [
    [0,     T_sea],
    [11000, T_sea - 0.0065 * 11000],
    [20000, T_sea - 0.0065 * 11000],   # constant in the lower stratosphere
]

env.set_atmospheric_model(
    type="custom_atmosphere",
    pressure=None,          # None = standard atmosphere pressure (correct for altitude)
    temperature=[[0, 311.7], [11000, 240.2], [25000, 240.2]], # ~35 °C at the pad
    wind_u=5,               # east component (m/s) — wind FROM 270° (west) blows east
    wind_v=0,               # north component (m/s)
)

# NOTE: the GFS forecast lines below are commented out on purpose.
# They would replace the June 2027 date and custom atmosphere above with
# tomorrow's real weather. Switch them back on ~1 week before launch.

# date set to tomorrow at noon for simulation purposes

import datetime

# tomorrow = datetime.date.today() + datetime.timedelta(days=1)

# env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12)) # setting date to tomorrow at noon

# setting atmospheric model to GFS forecast data

# env.set_atmospheric_model(type="Forecast", file="GFS")


# motor info from O6160.eng file
# (delete any "0 0" line at the start of the thrust data - it causes NaN errors)
# Masses come from the OpenRocket O6160 run:
#   loaded motor  = 53.683 kg (with motor) - 20.853 kg (no motor) = 32.83 kg
#   after burnout = 34.46 kg  ->  propellant = 53.68 - 34.46 = 19.22 kg
#   motor dry mass = 32.83 - 19.22 = 13.61 kg
# Motor runs ~165-287 cm from the nose tip in OpenRocket (~1.22 m long), so its
# center of mass sits ~0.61 m forward of the nozzle.
motor = GenericMotor.load_from_eng_file(
    "O6160.eng",
    propellant_initial_mass=19.22,
    dry_mass=13.61,
    chamber_height=1.22,
    chamber_position=0.61,
    center_of_dry_mass_position=0.61,
)
print("Propellant mass:", motor.propellant_initial_mass)
print("Dry mass:", motor.dry_mass)
print("Total impulse:", motor.total_impulse)
print("Burn time:", motor.burn_duration)
# motor.thrust.plot()   # uncomment to see the thrust curve (it pauses the script until closed)


# ===========================================================================
# ROCKET - Sockeye (from the PDR slides)
#
#   FROM OPENROCKET (O6160 config): length 292 cm, diameter 15.6 cm,
#                           mass without motor 20.853 kg, with motor 53.683 kg,
#                           CG 198 cm / CP 239 cm with motor, 2.58 cal stability
#   CONFIRMED from slides:  nosecone LV-Haack 30 in overall with a 6 in
#                           shoulder (24 in exposed)
#   CONFIRMED from fin slide: 4 fins, 11 in root, 4 in tip, 8 in span,
#                             7.5 in leading-edge sweep
#   FROM OPENROCKET:          drag curve (sockeye_drag.csv export),
#                             parachute sizes, 3.83 m rail, 4 deg launch angle
#
#   Values marked  ESTIMATE  are still assumed (mainly inertia, rail buttons,
#   boattail size and the exact fin position, measured off the OpenRocket drawing).
#   All positions are measured from the nose tip (nose_to_tail), in meters.
#   Airbrakes are assumed CLOSED for the whole flight.
# ===========================================================================

IN = 0.0254     # inches -> meters
LB = 0.45359    # pounds -> kilograms

ROCKET_LENGTH = 2.92                     # m (292 cm)
ROCKET_RADIUS = 0.156 / 2                # m (15.6 cm diameter)
ROCKET_MASS   = 20.853                   # kg, without motor (OpenRocket)

motor_loaded_mass = motor.dry_mass + motor.propellant_initial_mass
print(f"\nLoaded motor mass: {motor_loaded_mass:.3f} kg  ->  liftoff mass: {ROCKET_MASS + motor_loaded_mass:.3f} kg"
      f"  (OpenRocket: 53.683 kg)")

# Center of mass without motor, worked out from OpenRocket's export:
#   CG after burnout (rocket + empty motor, 34.46 kg) = 182.83 cm
#   with the empty motor's CG at ~226 cm:
#   (182.83 * 34.46 - 226 * 13.61) / 20.853 = ~154.6 cm
CG_WITHOUT_MOTOR = 1.546

# ESTIMATE: inertia treated as a uniform thin-walled tube (OpenRocket doesn't export it)
I_long = ROCKET_MASS * (6 * ROCKET_RADIUS**2 + ROCKET_LENGTH**2) / 12   # pitch / yaw
I_roll = ROCKET_MASS * ROCKET_RADIUS**2                                  # roll

# DRAG: from OpenRocket's O6160 "No wind, IREC site" export (or_no_wind.csv).
# OpenRocket gives a different Cd while the motor burns (no base drag) and
# after burnout, so the data is split at the BURNOUT event into a power-on
# and a power-off curve. Parachute data after apogee is ignored.

def read_openrocket_export(file_name):
    """Reads an OpenRocket CSV export. Returns (data table, {event: time})."""
    events, columns = {}, None
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# Time"):
                columns = [c.strip() for c in line[2:].split(",")]
            elif line.startswith("# Event"):
                parts = line.split()
                events.setdefault(parts[2], float(parts[5].replace("t=", "")))
    data = pd.read_csv(file_name, comment="#", header=None, names=columns)
    # shorter column names, e.g. "Altitude (m)" -> "Altitude"
    data.columns = [c.split(" (")[0] for c in data.columns]
    return data, events

def load_openrocket_drag(file_name):
    data, events = read_openrocket_export(file_name)
    ascent = data[(data["Time"] <= events["APOGEE"]) & (data["Mach number"] >= 0.02)]
    burning = ascent["Time"] <= events["BURNOUT"]

    def clean(part):
        # average repeated Mach values and sort from low to high Mach
        table = part.groupby(part["Mach number"].round(3))["Drag coefficient"].mean()
        return list(zip(table.index, table.values))

    return clean(ascent[burning]), clean(ascent[~burning])

DRAG_POWER_ON, DRAG_POWER_OFF = load_openrocket_drag("or_no_wind.csv")
print(f"Drag curve loaded: {len(DRAG_POWER_ON)} power-on points, {len(DRAG_POWER_OFF)} power-off points")

rocket = Rocket(
    radius=ROCKET_RADIUS,
    mass=ROCKET_MASS,
    inertia=(I_long, I_long, I_roll),
    power_off_drag=DRAG_POWER_OFF,
    power_on_drag=DRAG_POWER_ON,
    center_of_mass_without_motor=CG_WITHOUT_MOTOR,
    coordinate_system_orientation="nose_to_tail",
)

# motor nozzle sits at ~287 cm, just ahead of the short boattail (OpenRocket drawing)
rocket.add_motor(motor, position=2.87)

# nosecone: LV-Haack, 24 in exposed (30 in overall minus the 6 in shoulder)
rocket.add_nose(length=0.61, kind="lvhaack", position=0)

# fins: dimensions from the fin design slide
rocket.add_trapezoidal_fins(
    n=4,
    root_chord=11 * IN,
    tip_chord=4 * IN,
    span=8 * IN,
    sweep_length=7.5 * IN,        # leading edge sweeps back 7.5 in (11 in root - 4 in tip, straight trailing edge)
    position=2.58,                # ESTIMATE - root leading edge ~258 cm from the nose (OpenRocket drawing)
)

# ESTIMATE: short boattail at the aft end, ~5 cm long (OpenRocket drawing)
rocket.add_tail(top_radius=ROCKET_RADIUS, bottom_radius=0.071, length=0.05, position=2.87)

# ESTIMATE: rail buttons (one visible at ~160 cm in the OpenRocket drawing,
# the second assumed near the fin can)
rocket.set_rail_buttons(
    upper_button_position=1.60,
    lower_button_position=2.50,
    angular_position=45,
)

# recovery: sizes worked out from the OpenRocket drag export.
# OpenRocket reports Cd = 51.2 under the drogue and 886.1 under drogue + main,
# both relative to the body cross-section (15.6 cm diameter -> 0.01911 m^2):
#   drogue  cd_s = 51.231 * 0.01911              = ~0.98 m^2
#   main    cd_s = (886.111 - 51.231) * 0.01911  = ~15.96 m^2
# OpenRocket deploys the main at ~456 m (1,496 ft) AGL.
REF_AREA = np.pi * ROCKET_RADIUS**2
rocket.add_parachute("Drogue", cd_s=51.231 * REF_AREA, trigger="apogee", sampling_rate=105, lag=0)
rocket.add_parachute("Main", cd_s=(886.111 - 51.231) * REF_AREA, trigger=457,   # 457 m = 1500 ft AGL
                     sampling_rate=105, lag=0)

print(f"Static margin at liftoff: {rocket.static_margin(0):.2f} calibers (OpenRocket: 2.58 at Mach 0.3)")


# ===========================================================================
# IREC WEATHER CONDITIONS - Midland, TX, June 12-14
# Source: Weather Spark, "Average Weather in June in Midland, Texas"
#   https://weatherspark.com/m/4333/6/Average-Weather-in-June-in-Midland-Texas-United-States
#
# IMPORTANT: Weather Spark gives long-term CLIMATE AVERAGES, not a forecast.
# The page lists exact values for June 11 (93 °F / 69 °F) and June 16
# (94 °F / 70 °F); the June 12-14 numbers below are interpolated between
# those two days (~+0.2 °F per day).
#
# Site: 31.997 N, -102.078 W, elevation 2,776 ft (846 m) per Weather Spark
#       (our pad coords/elevation in the code: 32.0005 N, -102.0774 W, 848 m)
# Ground pressure (standard atmosphere at 848 m): ~91,540 Pa (915 hPa)
#
# Air density at the pad below is calculated from that pressure and the
# temperature (dry air, rho = p / (287.05 * T)). RocketPy computes density
# itself from the temperature + pressure we give it, so it is not an input,
# but it is printed below for each scenario as a check.
#
# ---------------------------------------------------------------------------
# JUNE 12
#   High temp:      ~93.2 °F (34.0 °C)   - typically mid/late afternoon (~4-5 PM)*
#   Low temp:       ~69.2 °F (20.7 °C)   - typically around sunrise (~6-7 AM)*
#   Average temp:   ~81.2 °F (27.3 °C)
#   Air density:    1.085 kg/m^3 at the low, 1.038 kg/m^3 at the high,
#                   ~1.061 kg/m^3 average
#   Wind:           ~11.9 mph (5.3 m/s) average, predominantly FROM THE SOUTH
#   Sunrise/sunset: 6:41 AM / 8:54 PM (earliest sunrise of the month)
#   Rain chance:    ~25% chance of a wet day
#   Cloud cover:    clear/mostly clear/partly cloudy ~79% of the time
#   Muggy chance:   ~14-15%
#
# JUNE 13
#   High temp:      ~93.4 °F (34.1 °C)   - typically mid/late afternoon (~4-5 PM)*
#   Low temp:       ~69.4 °F (20.8 °C)   - typically around sunrise (~6-7 AM)*
#   Average temp:   ~81.4 °F (27.4 °C)
#   Air density:    1.085 kg/m^3 at the low, 1.038 kg/m^3 at the high,
#                   ~1.061 kg/m^3 average
#   Wind:           ~11.9 mph (5.3 m/s) average, predominantly FROM THE SOUTH
#   Sunrise/sunset: ~6:41 AM / ~8:55 PM (not listed; between June 12 and 21)
#   Rain chance:    ~24% chance of a wet day
#   Cloud cover:    clear/mostly clear/partly cloudy ~79% of the time
#   Muggy chance:   ~15%
#
# JUNE 14
#   High temp:      ~93.6 °F (34.2 °C)   - typically mid/late afternoon (~4-5 PM)*
#   Low temp:       ~69.6 °F (20.9 °C)   - typically around sunrise (~6-7 AM)*
#   Average temp:   ~81.6 °F (27.6 °C)
#   Air density:    1.085 kg/m^3 at the low, 1.038 kg/m^3 at the high,
#                   ~1.061 kg/m^3 average
#   Wind:           ~11.8 mph (5.3 m/s) average, predominantly FROM THE SOUTH
#   Sunrise/sunset: ~6:41 AM / ~8:55 PM (not listed; between June 12 and 21)
#   Rain chance:    ~24% chance of a wet day
#   Cloud cover:    clear/mostly clear/partly cloudy ~78% of the time
#   Muggy chance:   ~15%
#
# *Weather Spark does not give the times highs/lows occur; these are the
#  normal daily pattern for West Texas in summer.
#
# ---------------------------------------------------------------------------
# JUNE 12-14 AVERAGES
#   High: ~93.4 °F (34.1 °C)    Low: ~69.4 °F (20.8 °C)    Mean: ~81.4 °F (27.4 °C)
#   Air density at pad: ~1.038 (hot) to ~1.085 (cool) kg/m^3
#   Wind: ~11.9 mph (5.3 m/s) from the south, very steady all month
#         (June average 11.7 mph, never more than 0.2 mph off)
#   Daily high range for June: rarely below 81 °F or above 102 °F
#   Daily low range for June:  rarely below 60 °F or above 77 °F
#
# WHAT THE CODE USES
#   Temperature  -> set_atmospheric_model(temperature=...)
#   Wind         -> wind_u / wind_v (wind FROM the south blows NORTH,
#                   so wind_v = +5.3, wind_u = 0)
#   Pressure     -> 915 mbar at the pad, decreasing with height consistently
#                   with the temperature profile (see atmosphere_profile)
#   Air density  -> calculated by RocketPy from temperature + pressure
#   Humidity, rain, clouds -> not modeled by RocketPy (very small effect
#                   on flight); listed above for launch planning only
#   Extreme case -> June record-range temps (60 °F / 102 °F) and a stronger
#                   wind (double the average) to see the worst case
# ===========================================================================

def F_to_C(F):
    return (F - 32) * 5 / 9

MPH_TO_MS = 0.44704

PAD_ELEVATION = 848       # m
PAD_PRESSURE = 91500      # Pa (915 mbar, same as the OpenRocket sims)

def atmosphere_profile(temp_C):
    """Temperature AND pressure vs height above sea level that agree with
    each other (hydrostatic balance), starting from temp_C and 915 mbar at
    the pad. Cools 6.5 °C/km up to 11 km, then constant temperature.
    Using standard-day pressure with a hot temperature makes the air too thin
    at altitude, so this keeps density realistic all the way up."""
    g, R, L = 9.80665, 287.05, 0.0065
    T_pad = temp_C + 273.15
    T_sea = T_pad + L * PAD_ELEVATION
    p_sea = PAD_PRESSURE * (T_sea / T_pad) ** (g / (R * L))
    heights = np.arange(0, 30001, 500)
    temps, pressures = [], []
    T_11 = T_sea - L * 11000
    p_11 = p_sea * (T_11 / T_sea) ** (g / (R * L))
    for h in heights:
        if h <= 11000:
            T = T_sea - L * h
            p = p_sea * (T / T_sea) ** (g / (R * L))
        else:
            T = T_11
            p = p_11 * np.exp(-g * (h - 11000) / (R * T_11))
        temps.append([h, T])
        pressures.append([h, p])
    return temps, pressures

def make_env(day, hour, temp_F, wind_mph, wind_from_deg=180):
    """Same Environment / set_date / set_atmospheric_model commands as above.
    wind_from_deg: direction the wind blows FROM (180 = south)."""
    env = Environment(latitude= 32.000507, longitude=-102.077408, elevation=PAD_ELEVATION)
    env.set_date((2027, 6, day, hour), timezone="America/Chicago")
    temperature, pressure = atmosphere_profile(F_to_C(temp_F))
    wind = wind_mph * MPH_TO_MS
    rad = np.radians(wind_from_deg)
    env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=pressure,
        temperature=temperature,
        wind_u=-wind * np.sin(rad),     # east component
        wind_v=-wind * np.cos(rad),     # north component
    )
    return env

# --- OpenRocket comparison cases (same settings as the team's .ork sims) ---
#   temperature 32.22 C (90 F), pressure 915 mbar at the pad, wind from 90 deg
#   medium wind 2.24 m/s (5 mph), high wind 6.71 m/s (15 mph)
#   OpenRocket's gusts/turbulence can't be copied - RocketPy wind is steady.
OR_TEMP_F = 32.22 * 9 / 5 + 32

#              label                          day hour  temp_F  wind_mph
scenarios = {
    "OR match: no wind":       make_env(13, 12, OR_TEMP_F, 0.0,  90),
    "OR match: medium wind":   make_env(13, 12, OR_TEMP_F, 2.24 / MPH_TO_MS, 90),
    "OR match: high wind":     make_env(13, 12, OR_TEMP_F, 6.71 / MPH_TO_MS, 90),
    "Jun 12 morning (69 °F)":   make_env(12,  7,  69.2,  11.9),
    "Jun 12 afternoon (93 °F)": make_env(12, 16,  93.2,  11.9),
    "Jun 13 morning (69 °F)":   make_env(13,  7,  69.4,  11.9),
    "Jun 13 afternoon (93 °F)": make_env(13, 16,  93.4,  11.9),
    "Jun 14 morning (70 °F)":   make_env(14,  7,  69.6,  11.8),
    "Jun 14 afternoon (94 °F)": make_env(14, 16,  93.6,  11.8),
    "Extreme hot + windy (102 °F, 24 mph)": make_env(13, 16, 102, 23.8),
}

# air density check (RocketPy's own calculation at the pad)
print("\nAir density at the pad for each scenario:")
for name, scenario_env in scenarios.items():
    print(f"  {name:40s} {scenario_env.density(scenario_env.elevation):.4f} kg/m^3")


# launch settings (from the OpenRocket sims)
RAIL_LENGTH = 3.83   # m - last year's rail. New 36 ft rail: change to 36 * 0.3048
INCLINATION = 86     # degrees from horizontal (OpenRocket: 4 deg from vertical)

# OpenRocket launches straight into the wind, so the heading is set to
# point at the direction each scenario's wind comes from.
WIND_FROM = {name: 180 for name in scenarios}
WIND_FROM.update({"OR match: no wind": 90, "OR match: medium wind": 90,
                  "OR match: high wind": 90,
                  "Extreme hot + windy (102 °F, 24 mph)": 180})


# running one flight per weather scenario

flights = {}
rows = []
for name, scenario_env in scenarios.items():
    flight = Flight(
        rocket=rocket,
        environment=scenario_env,
        rail_length=RAIL_LENGTH,
        inclination=INCLINATION,
        heading=WIND_FROM[name],
    )
    flights[name] = (flight, scenario_env)
    rows.append({
        "Scenario": name,
        "Pad air density (kg/m^3)": round(scenario_env.density(scenario_env.elevation), 4),
        "Apogee AGL (m)": round(flight.apogee - scenario_env.elevation, 3),
        "Apogee AGL (ft)": round((flight.apogee - scenario_env.elevation) * 3.281, 3),
        "Apogee time (s)": round(flight.apogee_time, 3),
        "Max speed (m/s)": round(flight.max_speed, 3),
        "Max Mach": round(flight.max_mach_number, 3),
        "Rail exit speed (m/s)": round(flight.out_of_rail_velocity, 3),
        "Landing drift (m)": round(np.hypot(flight.x_impact, flight.y_impact), 3),
    })

summary = pd.DataFrame(rows)
summary.to_csv("sockeye_weather_summary.csv", index=False)
print(summary.to_string(index=False))

# --- check against OpenRocket's three IREC-site runs (O6160 exports) ---
OR_FILES = {
    "OR match: no wind":     "or_no_wind.csv",
    "OR match: medium wind": "or_medium_wind.csv",
    "OR match: high wind":   "or_high_wind.csv",
}
or_runs = {name: read_openrocket_export(file) for name, file in OR_FILES.items()}

def openrocket_results(data, events):
    ascent = data[data["Time"] <= events["APOGEE"]]
    rail_exit = data.iloc[(data["Time"] - events["LAUNCHROD"]).abs().argmin()]
    return {"Apogee AGL (m)": ascent["Altitude"].max(),
            "Apogee time (s)": events["APOGEE"],
            "Max speed (m/s)": ascent["Total velocity"].max(),
            "Max Mach": ascent["Mach number"].max(),
            "Rail exit speed (m/s)": rail_exit["Total velocity"],
            "Landing time (s)": events["GROUND_HIT"],
            "Landing drift (m)": data["Lateral distance"].iloc[-1]}

def rocketpy_results(flight, scenario_env):
    return {"Apogee AGL (m)": flight.apogee - scenario_env.elevation,
            "Apogee time (s)": flight.apogee_time,
            "Max speed (m/s)": flight.max_speed,
            "Max Mach": flight.max_mach_number,
            "Rail exit speed (m/s)": flight.out_of_rail_velocity,
            "Landing time (s)": flight.t_final,
            "Landing drift (m)": np.hypot(flight.x_impact, flight.y_impact)}

comparison_rows = []
for name, (or_data, or_events) in or_runs.items():
    rp = rocketpy_results(*flights[name])
    orr = openrocket_results(or_data, or_events)
    print(f"\nRocketPy vs OpenRocket - {name.replace('OR match: ', '')}:")
    for key in rp:
        diff = 100 * (rp[key] - orr[key]) / orr[key]
        print(f"  {key:22s} RocketPy {rp[key]:10.3f}   OpenRocket {orr[key]:10.3f}   ({diff:+.1f}%)")
        comparison_rows.append({"Scenario": name, "Quantity": key,
                                "RocketPy": round(rp[key], 3), "OpenRocket": round(orr[key], 3),
                                "Difference (%)": round(diff, 1)})
print(f"\n  Motor burnout: RocketPy {motor.burn_out_time:.3f} s, "
      f"OpenRocket {or_runs['OR match: no wind'][1]['BURNOUT']:.3f} s")
print("  Note: landing drift depends on OpenRocket's gusts, which RocketPy's steady wind can't copy.")

pd.DataFrame(comparison_rows).to_csv("sockeye_vs_openrocket.csv", index=False)


# ---------------------------------------------------------------------------
# Plots - motion of the rocket for every scenario on the same axes
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for name, (flight, scenario_env) in flights.items():
    z = flight.z.source                 # columns: time, altitude above sea level
    axes[0, 0].plot(z[:, 0], z[:, 1] - scenario_env.elevation, label=name)

    v = flight.speed.source
    axes[0, 1].plot(v[:, 0], v[:, 1], label=name)

    a = flight.acceleration.source
    axes[1, 0].plot(a[:, 0], a[:, 1], label=name)

    x, y = flight.x.source, flight.y.source
    axes[1, 1].plot(x[:, 1], y[:, 1], label=name)

axes[0, 0].set(title="Altitude vs time", xlabel="Time (s)", ylabel="Altitude AGL (m)")
axes[0, 1].set(title="Speed vs time", xlabel="Time (s)", ylabel="Speed (m/s)")
axes[1, 0].set(title="Acceleration vs time", xlabel="Time (s)", ylabel="Acceleration (m/s²)")
axes[1, 1].set(title="Ground track (top view)", xlabel="East of pad (m)", ylabel="North of pad (m)")
axes[1, 1].set_aspect("equal", adjustable="datalim")
for ax in axes.flat:
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)

fig.suptitle("Sockeye - simulated flights at IREC (Midland, TX)", fontsize=14)
plt.tight_layout()
plt.savefig("sockeye_motion.png", dpi=150)

# RocketPy (solid) vs OpenRocket (dashed) for the 3 comparison cases:
# altitude, vertical velocity and vertical acceleration vs time
fig3, axes3 = plt.subplots(3, 3, figsize=(18, 12), sharex="col")
for col, name in enumerate(OR_FILES):
    flight, scenario_env = flights[name]
    or_data, or_events = or_runs[name]
    z, vz, az = flight.z.source, flight.vz.source, flight.az.source
    rows = [
        ("Altitude AGL (m)", z[:, 0], z[:, 1] - scenario_env.elevation, "Altitude"),
        ("Vertical velocity (m/s)", vz[:, 0], vz[:, 1], "Vertical velocity"),
        ("Vertical acceleration (m/s²)", az[:, 0], az[:, 1], "Vertical acceleration"),
    ]
    for row, (label, t, y, or_col) in enumerate(rows):
        ax = axes3[row, col]
        ax.plot(t, y, label="RocketPy")
        ax.plot(or_data["Time"], or_data[or_col], "--", label="OpenRocket")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(name.replace("OR match: ", "").capitalize())
            ax.legend()
        if row == 2:
            ax.set_xlabel("Time (s)")
            ax.set_xlim(0, 60)          # zoom on launch to just past apogee (full flight is in sockeye_motion.png)
            ax.set_ylim(-80, 130)
fig3.suptitle("Sockeye: RocketPy vs OpenRocket (O6160, IREC conditions)", fontsize=14)
plt.tight_layout()
plt.savefig("sockeye_vs_openrocket.png", dpi=150)

# zoom in on the motor burn / boost phase
fig2, ax2 = plt.subplots(figsize=(8, 4.5))
baseline_flight = flights["Jun 13 afternoon (93 °F)"][0]
v = baseline_flight.vz.source
burn_end = motor.burn_out_time
mask = v[:, 0] <= burn_end * 2
ax2.plot(v[mask, 0], v[mask, 1])
ax2.axvline(burn_end, linestyle="--", color="gray", label="Motor burnout")
ax2.set(title="Vertical velocity during boost (Jun 13 afternoon)", xlabel="Time (s)", ylabel="Vz (m/s)")
ax2.grid(alpha=0.3)
ax2.legend()
plt.tight_layout()
plt.savefig("sockeye_boost.png", dpi=150)

# RocketPy full built-in report for the Jun 13 afternoon flight (stability, forces, etc.)
# baseline_flight.all_info()

# show every graph window at once, after all the simulations have finished
plt.show()
