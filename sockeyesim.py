from rocketpy import Environment, SolidMotor, Rocket, Flight, GenericMotor 
import matplotlib
from rocketpy import GenericMotor
from ork_loader import load_rocket_from_ork



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

# date set to tomorrow at noon for simulation purposes

import datetime 

tomorrow = datetime.date.today() + datetime.timedelta(days=1)

env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12)) # setting date to tomorrow at noon

# setting atmospheric model to GFS forecast data

env.set_atmospheric_model(type="Forecast", file="GFS")

# env.info()

# motor info from O6160.eng file 
motor = GenericMotor.load_from_eng_file("O6160.eng")
print("Propellant mass:", motor.propellant_initial_mass)
print("Dry mass:", motor.dry_mass)
print("Total impulse:", motor.total_impulse)
print("Burn time:", motor.burn_duration)

rocket, params = load_rocket_from_ork(
    "sockeye_pdr.ork",
    ork_jar="OpenRocket-23.09.jar",
    motor=motor
)
