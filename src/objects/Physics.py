import json
import math

# constants at sea level
ATM_SEA_LEVEL = {
    'm_mol': 28.966 * 1e-3,
    'pressure': 101325,
    'temperature': 288.15,
    'tz': -6.5 * 1e-3,
    'density': 1.225,
    'viscosity': 1.458 * 1e-6,
    'dynamic_visco': 1.8e-5,
    'gamma': 1.4,
    'R': 8.314,
    'Cp': ((8.314 / (28.966 * 1e-3)) * 1.4) / (1.4 - 1),
    'Cv': (8.314 / (28.966 * 1e-3)) / (1.4 - 1),
    'gravity': 9.80665,
    'altitude': 0,
}

# meters
ATM_LAYER_ALT = {
    'z_troposphere': 11019,
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
        if _z < ATM_LAYER_ALT['z_troposphere']:
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

class Pressure:
    def __init__(self):
        # TODO: change the constant of the earth pressures
        # goal => be able to do calculation on other planets :)
        self.troposphere = lambda z, g, rs: ATM_SEA_LEVEL['pressure'] * (1 + z * (ATM_SEA_LEVEL['tz'] / ATM_SEA_LEVEL['temperature'])) ** (-g / (rs * ATM_SEA_LEVEL['tz']))
        self.above = lambda z: 22632 * math.exp(-(z - 11019) / 6341.6)

    def get_pressure(self, _z, _g, _rs):
        if _z < ATM_LAYER_ALT['z_troposphere']:
            return self.troposphere(_z, _g, _rs)
        else:
            return self.above(_z)

class Atmosphere:
    def __init__(self):
        # initial state
        self.m_mol          = ATM_SEA_LEVEL['m_mol']
        self.pressure       = ATM_SEA_LEVEL['pressure']
        self.temperature    = ATM_SEA_LEVEL['temperature']
        self.density        = ATM_SEA_LEVEL['density']
        self.viscosity      = ATM_SEA_LEVEL['viscosity']
        self.t_evolution    = Temperature()
        self.p_evolution    = Pressure()
        self.gamma          = ATM_SEA_LEVEL['gamma']
        self.R              = ATM_SEA_LEVEL['R']
        self.altitude       = ATM_SEA_LEVEL['altitude']

        self.rs = self.R / self.m_mol
        self.cp = (self.rs * self.gamma) / (self.gamma - 1)
        self.cv = self.rs / (self.gamma - 1)

    def get_atm_at_z(self, _z, _gravity):
        delta_z = _z - self.altitude

        self.pressure = self.p_evolution.get_pressure(_z, _gravity, self.rs)
        self.temperature = self.t_evolution.get_temperature(_z)
        self.density = self.pressure / (self.temperature * self.rs)
        self.viscosity = self.viscosity * (self.temperature ** (3 / 2) / (110.4 + self.temperature))

        self.altitude   = _z

        return {'pressure': self.pressure, 'temperature': self.temperature,
                'density': self.density, 'viscosity': self.viscosity}

    def get_atm_data(self):
        return {'m_mol': self.m_mol, 'pressure': self.pressure, 'temperature': self.temperature, 'density': self.density,
                'viscosity': self.viscosity, 'gamma': self.gamma, 'R': self.R, 'rs': self.rs,
                'cp': self.cp, 'cv': self.cv}

    def from_dict(self, _dict):
        if _dict['class'] == 'Atmosphere':
            self.m_mol          = _dict['m_mol']
            self.pressure       = _dict['pressure']
            self.temperature    = _dict['temperature']
            self.density        = _dict['density']
            self.viscosity      = _dict['viscosity']
            self.gamma          = _dict['gamma']
            self.R              = _dict['R']
            self.altitude       = _dict['altitude']


    def to_dict(self):
        return {'class':        'Atmosphere',
                'm_mol':        self.m_mol,
                'pressure':     self.pressure,
                'temperature':  self.temperature,
                'density':      self.density,
                'viscosity':    self.viscosity,
                'gamma':        self.gamma,
                'R':            self.R,
                'altitude':     self.altitude,
                }

    def to_json(self):
        return json.dumps(self.to_dict())



class Physics:
    def __init__(self):
        self.atm = Atmosphere()
        self.gravity = ATM_SEA_LEVEL['gravity']
        self.velocity_x = 0
        self.velocity_y = 0

    def from_dict(self, _dict):
        if _dict['class'] == 'Physics':
            self.atm.from_dict(_dict['Atm'])

            self.gravity    = _dict['gravity']
            self.velocity_x = _dict['velocity_x']
            self.velocity_y = _dict['velocity_y']

    def to_dict(self):
        return {'class'     : 'Physics',
                'Atm'       : self.atm.to_dict(),
                'gravity'   : self.gravity,
                'velocity_x': self.velocity_x,
                'velocity_y': self.velocity_y,
                }

    def to_json(self):
        return json.dumps(self.to_dict())

    def get_rho(self):
        return self.atm.pressure / (self.atm.rs * self.atm.temperature)

    def get_mu(self):
        return (self.atm.temperature / ATM_SEA_LEVEL['temperature']) ** (3 / 2) * (
                    (ATM_SEA_LEVEL['temperature'] + 110) / (self.atm.temperature + 110)) * ATM_SEA_LEVEL['dynamic_visco']

    def get_local_reynolds(self, _x):
        # TODO: velocity y
        return self.get_rho() * self.velocity_x * _x * 1e-3 / self.get_mu()




