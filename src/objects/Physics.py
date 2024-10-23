from unittest import TestLoader

import numpy as np

# constants at sea level
ATM_SEA_LEVEL = {
    'm_mol': 28.966 * 1e-3,
    'pressure': 101325,
    'temperature': 288.15,
    'tz': -6.5 * 1e-3,
    'density': 1.225,
    'viscosity': 1.458 * 1e-6,
    'gamma': 1.4,
    'R': 8.314,
    'Cp': ((8.314 / (28.966 * 1e-3)) * 1.4) / (1.4 - 1),
    'Cv': (8.314 / (28.966 * 1e-3)) / (1.4 - 1),
    'gravity': 9.80665,
    'altitude': 0,
}

# meters
ATM_LAYER_ALT = {
    'z_tropopause': 11019,
    'z_stratosphere1': 20063,
    'z_stratosphere2': 32162,
    'z_stratopause': 47350,
    'z_mesosphere1': 51413,
    'z_mesosphere2': 71802,
    'z_mesopause': 86000
}

# Data from wikipedia : https://fr.wikipedia.org/wiki/Atmosph%C3%A8re_normalis%C3%A9e
class Temperature:
    def __init__(self):
        self.tz_troposphere = lambda x: 288.15 - 6.5 * 1e-3 * x
        self.tz_tropopause = lambda x: 216.65
        self.tz_stratosphere1 = lambda x: 216.65 + 1e-3 * (x - ATM_LAYER_ALT['z_stratosphere1'])
        self.tz_stratosphere2 = lambda x: 228.65 + 2.8 * 1e-3 * (x - ATM_LAYER_ALT['z_stratosphere2'])
        self.tz_stratopause = lambda x: 270.65
        self.tz_mesosphere1 = lambda x: 270.65 - 2.8 * 1e-3 * (x - ATM_LAYER_ALT['z_mesosphere1'])
        self.tz_mesosphere2 = lambda x: 214.65 - 2 * 1e-3 * (x - ATM_LAYER_ALT['z_mesosphere2'])
        self.tz_mesopause = lambda x: 186.95

    def get_temperature(self, _z):
        if _z < ATM_LAYER_ALT['z_tropopause']:
            return self.tz_troposphere(_z)
        elif _z < ATM_LAYER_ALT['z_stratosphere1']:
            return self.tz_tropopause(_z)
        elif _z < ATM_LAYER_ALT['z_stratosphere2']:
            return self.tz_stratosphere1(_z)
        elif _z < ATM_LAYER_ALT['z_stratopause']:
            return self.tz_stratosphere2(_z)
        elif _z < ATM_LAYER_ALT['z_mesosphere1']:
            return self.tz_stratopause(_z)
        elif _z < ATM_LAYER_ALT['z_mesosphere2']:
            return self.tz_mesosphere1(_z)
        elif _z < ATM_LAYER_ALT['z_mesopause']:
            return self.tz_mesosphere2(_z)
        else:
            return self.tz_mesopause(_z)


class Air:
    def __init__(self):
        # initial state
        self.m_mol          = ATM_SEA_LEVEL['m_mol']
        self.pressure       = ATM_SEA_LEVEL['pressure']
        self.temperature    = ATM_SEA_LEVEL['temperature']
        self.density        = ATM_SEA_LEVEL['density']
        self.viscosity      = ATM_SEA_LEVEL['viscosity']
        self.tz             = ATM_SEA_LEVEL['tz']
        self.t_evolution    = Temperature()
        self.gamma          = ATM_SEA_LEVEL['gamma']
        self.R              = ATM_SEA_LEVEL['R']
        self.gravity        = ATM_SEA_LEVEL['gravity']
        self.altitude       = ATM_SEA_LEVEL['altitude']

        self.rs = self.R / self.m_mol
        self.cp = (self.rs * self.gamma) / (self.gamma - 1)
        self.cv = self.rs / (self.gamma - 1)

    def get_atm_at_z(self, _z):
        delta_z = _z - self.altitude

        # TODO: change the pressure formula -> It goes negative between 40k and 50k
        self.pressure = self.pressure * (1 + (self.tz / self.temperature) * delta_z) ** (-self.gravity / (self.tz * self.rs))
        self.temperature = self.t_evolution.get_temperature(_z)
        self.density = self.pressure / (self.temperature * self.rs)
        self.viscosity = self.viscosity * (self.temperature ** (3 / 2) / (110.4 + self.temperature))

        self.altitude = _z

        return {'pressure': self.pressure, 'temperature': self.temperature,
                'density': self.density, 'viscosity': self.viscosity}

    def get_atm_data(self):
        return {'m_mol': self.m_mol, 'pressure': self.pressure, 'temperature': self.temperature, 'density': self.density,
                'viscosity': self.viscosity, 'tz': self.tz, 'gamma': self.gamma, 'R': self.R, 'gravity': self.gravity,
                'rs': self.rs, 'cp': self.cp, 'cv': self.cv}


class Shock:
    def __init__(self, _z=0):
        self.air = Air()
        if _z != 0:
            self.air.get_atm_at_z(_z)

    def get_mach_number(self, _flow_velocity):
        sound_speed = np.sqrt(self.air.gamma * self.air.rs * self.air.temperature)
        return _flow_velocity / sound_speed


