from scipy.optimize import newton
from scipy.integrate import simpson
from src.objects.JsonObject import JsonObject
from src.objects.Profile import *
import numpy as np

# constants at sea level
ATM_SEA_LEVEL = {
    'm_mol': 28.966 * 1e-3,
    'pressure': 101325,
    'temperature': 288.15,
    'tz': -6.5 * 1e-3,
    'density': 1.225,
    'viscosity': 1.458 * 1e-6,
    'mu': 1.8e-5,
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

class Atmosphere(JsonObject):
    def __init__(self):
        super().__init__()
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
        self.mu             = ATM_SEA_LEVEL['mu']

        self.rs = self.R / self.m_mol
        self.cp = (self.rs * self.gamma) / (self.gamma - 1)
        self.cv = self.rs / (self.gamma - 1)

    def get_atm_at_z(self, _z, _gravity):
        delta_z = _z - self.altitude

        self.pressure = self.p_evolution.get_pressure(_z, _gravity, self.rs)
        self.temperature = self.t_evolution.get_temperature(_z)
        self.density = self.pressure / (self.temperature * self.rs)
        self.viscosity = self.viscosity * (self.temperature ** (3 / 2) / (110.4 + self.temperature))
        self.mu = (self.temperature / ATM_SEA_LEVEL['temperature']) ** (3 / 2) * ((ATM_SEA_LEVEL['temperature'] + 110) / self.temperature + 110) * ATM_SEA_LEVEL['mu']

        self.altitude   = _z

        return {'pressure': self.pressure, 'temperature': self.temperature,
                'density': self.density, 'viscosity': self.viscosity}
    def get_atm_data(self):
        return {'m_mol': self.m_mol, 'pressure': self.pressure, 'temperature': self.temperature, 'density': self.density,
                'viscosity': self.viscosity, 'gamma': self.gamma, 'R': self.R, 'rs': self.rs,
                'cp': self.cp, 'cv': self.cv}

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



class Physics(JsonObject):
    def __init__(self):
        super().__init__()
        self.atm = Atmosphere()
        self.gravity = ATM_SEA_LEVEL['gravity']
        self.velocity_x = 0
        self.velocity_y = 0


    def to_dict(self):
        return {'class'     : 'Physics',
                'atm'       : self.atm.to_dict(),
                'gravity'   : self.gravity,
                'velocity_x': self.velocity_x,
                'velocity_y': self.velocity_y,
                }

    # overwrite the from_dict function from JsonObject to handle the atm variable
    def from_dict(self, _dict):
        # Deserialize Atmosphere separately if present
        if 'atm' in _dict:
            self.atm.from_dict(_dict['atm'])  # Ensure atm is correctly deserialized as Atmosphere object
            _dict.pop('atm')
        super().from_dict(_dict) # Call the base's function





class HypersonicObliqueShock(JsonObject):
    def __init__(self, _physic=Physics(), _profile=Profile()):
        super().__init__()
        self.physic : Physics = _physic # Physic class
        self.profile : Profile = _profile # Profile class

        self.section_shape = np.array([])
        self.section_method = {}

        self.sound_speed = 0

        self.mach_inf = 0
        self.theta = np.array([])
        self.beta = np.array([])

        self.x_shock_curve = np.array([])
        self.y_shock_curve = np.array([])

        self.flow_characteristics = {}

        self.upstream_var_stag = {}
        self.downstream_var_stag = {}

        self.pressure_coeff = np.array([])
        self.drag_coeff = np.array([])
        self.lift_coeff = np.array([])

        self.stand_off_distance_arr = []

    def calcul(self):
        self.section_shape = np.concatenate(
            ([0], np.cumsum(np.array([len(section['x']) for section in self.profile.get_section().values()]))))
        self.section_method = {}
        for index, section_name in enumerate(self.profile.get_section().keys()):
            self.section_method[section_name] = {}
            self.section_method[section_name]["interval"] = (self.section_shape[index], self.section_shape[index + 1])

        self.sound_speed = np.sqrt(self.physic.atm.gamma * self.physic.atm.rs * self.physic.atm.temperature)

        # get infinite mach number and flow angles
        self.mach_inf = self.get_mach_number()
        self.theta, self.beta = self.get_deviation_and_shock_angle()

        # get the method for each section
        self.section_method = self.get_method(0.98)

        # shock layer
        self.x_shock_curve, self.y_shock_curve = self.get_shock_layer()

        # get flow variable
        self.flow_characteristics = self.get_flow_characteristics()

        # stagnation variable upstream and downstream
        self.upstream_var_stag, self.downstream_var_stag = self.stagnation()

        # pressure coefficient variation
        self.pressure_coeff, self.drag_coeff, self.lift_coeff = self.coefficient()

    def to_dict(self):
        # Convert all numpy variable into serializable value (e.g. python type)
        self.flow_characteristics = {key: value.tolist() for key, value in self.flow_characteristics.items()}

        self.x_shock_curve = list(map(float, self.x_shock_curve))
        self.y_shock_curve = list(map(float, self.y_shock_curve))
        self.section_shape = list(map(int, self.section_shape))

        self.section_method = {
            key: {
                'interval': tuple(int(x) for x in value['interval']),
                'method': value['method']
            } if 'interval' in value else value
            for key, value in self.section_method.items()
        }

        self.theta = list(map(float, self.theta))
        self.beta = list(map(float, self.beta))

        self.upstream_var_stag = {key: value.tolist() for key, value in self.upstream_var_stag.items()}
        self.downstream_var_stag = {key: value.tolist() for key, value in self.downstream_var_stag.items()}

        self.pressure_coeff = list(map(float, self.pressure_coeff))
        self.drag_coeff = list(map(float, self.drag_coeff))
        self.lift_coeff = list(map(float, self.lift_coeff))


        return {'class': 'HypersonicObliqueShock',
                'physic': self.physic.to_dict(),
                'profile': self.profile.to_dict(),
                'section_shape': self.section_shape,
                'section_method': self.section_method,
                'sound_speed': int(self.sound_speed),
                'mach_inf': int(self.mach_inf),
                'theta': self.theta,
                'beta': self.beta,
                'x_shock_curve': self.x_shock_curve,
                'y_shock_curve': self.y_shock_curve,
                'flow_characteristics': self.flow_characteristics,
                'upstream_var_stag': self.upstream_var_stag,
                'downstream_var_stag': self.downstream_var_stag,
                'pressure_coeff': self.pressure_coeff,
                'drag_coeff': self.drag_coeff,
                'lift_coeff': self.lift_coeff,
                'stand_off_distance_arr': self.stand_off_distance_arr
                }

    # overwrite JsonObject from_dict function to handle physic and profile variable
    def from_dict(self, _dict):
        if 'physic' in _dict:
            self.physic.from_dict(_dict['physic'])
            _dict.pop('physic')
        if 'profile' in _dict:
            self.profile.from_dict(_dict['profile'])
            _dict.pop('profile')

        super().from_dict(_dict) # Call the base's function

        # Convert all python variable into numpy value
        self.flow_characteristics = {key: np.array(value) for key, value in self.flow_characteristics.items()}

        self.x_shock_curve = np.array(self.x_shock_curve, dtype=np.float64)
        self.y_shock_curve = np.array(self.y_shock_curve, dtype=np.float64)
        self.section_shape = np.array(self.section_shape, dtype=np.int64)

        self.section_method = {
            key: {
                'interval': np.array([int(x) for x in value['interval']], dtype=np.int64),
                'method': value['method']
            } if 'interval' in value else value
            for key, value in self.section_method.items()
        }

        self.theta = np.array(self.theta, dtype=np.float64)
        self.beta = np.array(self.beta, dtype=np.float64)

        self.upstream_var_stag = {key: np.array(value) for key, value in self.upstream_var_stag.items()}
        self.downstream_var_stag = {key: np.array(value) for key, value in self.downstream_var_stag.items()}

        self.pressure_coeff = np.array(self.pressure_coeff, dtype=np.float64)
        self.drag_coeff = np.array(self.drag_coeff, dtype=np.float64)
        self.lift_coeff = np.array(self.lift_coeff, dtype=np.float64)

    def get_mach_number(self):
        return self.physic.velocity_x / self.sound_speed

    def get_local_reynolds(self, _x):
        # TODO: velocity y
        return self.flow_characteristics['density'] * self.flow_characteristics['velocity_n'] * _x / self.physic.atm.mu

    def get_boundary_layer(self, _x):
        r = 0.85 # TODO: check the reynolds > 3000 for the turbulent => 0.9 turbulent 0.85 laminar
        tf = 1 + r * ((self.physic.atm.gamma - 1) / 2) * (self.flow_characteristics['mach_amb'] ** 2)
        tbar = tf * self.flow_characteristics['temperature']
        # Chapman-Rubesin constant
        c0 = (self.physic.atm.mu / ATM_SEA_LEVEL['mu']) / (tbar / self.flow_characteristics['temperature'])
        return _x * np.sqrt(c0) * ((self.physic.atm.gamma - 1) / 2) * (
                    self.flow_characteristics['mach_amb'] ** 2) / np.sqrt(self.get_local_reynolds(_x))


    def get_deviation_and_shock_angle(self):
        dy_dx = np.gradient(self.profile.get_y(), self.profile.get_x())
        dy_dx[np.abs(dy_dx) < 1e-10] = 0

        theta = np.arctan(dy_dx)
        beta = np.zeros_like(theta)

        def cotangent(_angle):
            return np.cos(_angle) / np.sin(_angle)

        def beta_equation(_beta, _mach, _theta, _gamma):
            left_side = np.tan(_theta)
            right_side = 2 * cotangent(_beta) * ((_mach**2 * np.sin(_beta)**2 - 1) / (_mach ** 2 * (_gamma + np.cos(2 * _beta)) + 2))
            return left_side - right_side

        for index, theta_i in enumerate(theta):
            try:
                beta_init = np.radians(30)
                beta_newton = newton(beta_equation, beta_init, args=(self.mach_inf, theta_i, self.physic.atm.gamma))
                beta[index] = beta_newton
            except RuntimeError:
                beta[index] = np.nan

        return theta, beta

    def get_method(self, _tol):
        mask_neg = self.theta < 0
        neg_arr = np.where(mask_neg == True)[0]

        if np.any(neg_arr):
            idx_start = np.searchsorted(self.section_shape, neg_arr[0], side='right')
            idx_end = np.searchsorted(self.section_shape, neg_arr[-1], side='right')

            if idx_start == idx_end:
                interval = (self.section_shape[idx_start -1], self.section_shape[idx_end])

                for section_name in self.profile.get_section().keys():
                    if self.section_method[section_name]["interval"] == interval:
                        self.section_method[section_name]["method"] = "Prandtl-Meyer"

        for section_name in self.profile.get_section().keys():
            interval = self.section_method[section_name]["interval"]
            theta_section = self.theta[interval[0]:interval[1]]

            ref_value = np.median(theta_section)
            close_values = np.isclose(theta_section, ref_value, rtol=1e-5)
            ratio_close = np.sum(close_values) / len(theta_section)

            if np.isclose(ref_value, 0):
                self.section_method[section_name]['method'] = "Rankine-Hugoniot"
            elif np.all(theta_section >= 0) and ratio_close >= _tol:
                self.section_method[section_name]['method'] = "Diedre"
            elif np.all(theta_section >= 0):
                self.section_method[section_name]['method'] = "Rankine-Hugoniot"
            else:
                self.section_method[section_name]['method'] = None

        return self.section_method



    def get_shock_layer(self):
        def delta(_diameter):
            return _diameter * 0.193 * np.exp(4.67 / self.mach_inf ** 2)

        x_shock_curve, y_shock_curve = [np.zeros(len(self.profile.get_x())) for _ in range(2)]

        x_shock_curve[x_shock_curve == 0.0] = np.nan
        y_shock_curve[y_shock_curve == 0.0] = np.nan

        for section_name, section_values in self.profile.get_section().items():
            interval = self.section_method[section_name]['interval']
            beta_section = self.beta[interval[0]:interval[1]]

            if np.any(np.isnan(beta_section)):
                first_finite_index = np.where(np.isfinite(beta_section))[0]
                if first_finite_index[0] != 0:
                    first_finite_value = beta_section[first_finite_index[0]]
                    beta_section[:first_finite_index[0]] = first_finite_value
                else:
                    second_finite_value = beta_section[first_finite_index[1]]
                    beta_section[:first_finite_index[1]] = second_finite_value

                if 'radius' in section_values:
                    stand_off_distance = delta(_diameter=2 * section_values['radius'])
                    if self.section_method[section_name]['interval'][0] == 0:
                        x_start = -stand_off_distance
                        y_start = 0
                    else:
                        x_start = np.min(section_values['x']) - stand_off_distance
                        y_start = np.min(section_values['y'])

                    self.stand_off_distance_arr.append(stand_off_distance)

                    x_end = np.max(section_values['x'])
                    y_end = np.max(section_values['y']) + stand_off_distance

                    section_length = np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
                    point_distance = section_length / len(beta_section)

                    x_section_curve = [x_start]
                    y_section_curve = [y_start]

                    for beta in beta_section:
                        x_section_curve.append(x_section_curve[-1] + point_distance * np.cos(beta))
                        y_section_curve.append(y_section_curve[-1] + point_distance * np.sin(beta))

                    x_shock_curve[interval[0]:interval[1]] = x_section_curve[:-1]
                    y_shock_curve[interval[0]:interval[1]] = y_section_curve[:-1]
            elif self.section_method[section_name]['method'] == "Diedre":
                length = np.max(section_values['x']) - np.min(section_values['x'])
                x_diedre = np.linspace(0, length, len(section_values['x']))
                y_diedre = np.tan(beta_section) * x_diedre + np.min(section_values['y'])

                x_shock_curve[interval[0]:interval[1]] = section_values['x']
                y_shock_curve[interval[0]:interval[1]] = y_diedre

            else:
                expansion_value = 1.3
                x_shock_curve[interval[0]:interval[1]] = section_values['x']
                y_shock_curve[interval[0]:interval[1]] = section_values['y'] * expansion_value

        return x_shock_curve, y_shock_curve

    def variable_downstream_of_shock(self, _mach):
        gamma = self.physic.atm.gamma

        pressure = ((2 * gamma * _mach ** 2) / (gamma + 1) - ((gamma - 1) / (gamma + 1))) * self.physic.atm.pressure
        temperature = np.divide(np.multiply((1 + 0.5 * (gamma - 1) * _mach ** 2), ((2 * gamma) / (gamma - 1)) * _mach ** 2 - 1),
                                _mach ** 2 * np.divide((gamma + 1) ** 2, 2 * (gamma - 1))) * self.physic.atm.temperature
        density = (((gamma + 1) * _mach ** 2) / ((gamma - 1) * _mach ** 2 + 2)) * self.physic.atm.density
        mach_n = np.sqrt(np.divide((gamma - 1) * _mach ** 2 + 2, 2 * gamma * _mach ** 2 - (gamma - 1)))
        soundspeed_n = np.sqrt(np.divide(temperature, self.physic.atm.temperature * self.sound_speed))

        return {
            'pressure': pressure,
            'temperature': temperature,
            'density': density,
            'mach_n': mach_n,
            'soundspeed_n': soundspeed_n,
            'velocity_n': mach_n * soundspeed_n
        }

    def prandtl_meyer_variable(self, _mach, _interval, _pressure_up, _temperature_up):
        def prandtl_meyer_fnc(_mach):
            lambda_ = np.sqrt((self.physic.atm.gamma - 1) / (self.physic.atm.gamma + 1))
            return (1 / lambda_) * np.arctan(lambda_ * np.sqrt(_mach ** 2 - 1)) - np.arctan(np.sqrt(_mach ** 2 - 1))

        def inv_prandtl_meyer(_mach_i, _nu_target):
            nu_current = prandtl_meyer_fnc(_mach_i)
            return nu_current - _nu_target

        def pressure_ratio(_mach_up, _mach_down):
            gamma = self.physic.atm.gamma
            return np.divide(1 + 0.5 * (gamma - 1) * _mach_up ** 2, 1 + 0.5 * (gamma - 1) * _mach_down ** 2) ** (-gamma / (gamma - 1))

        def temperature_ratio(_mach_up, _mach_down):
            gamma = self.physic.atm.gamma
            return (1 + 0.5 * (gamma - 1) * _mach_down ** 2) / (1 + 0.5 * (gamma - 1)) * _mach_up ** 2

        def rho(_pressure, _temperature):
            # TODO: check the value of rs if it R or rs
            return _pressure / (self.physic.atm.rs * _temperature)

        # expression de mach au point d'impact
        gamma = self.physic.atm.gamma
        mach_n_c = np.sqrt(np.divide((gamma - 1) * _mach ** 2 + 2, 2 * gamma * _mach ** 2 - (gamma - 1)))
        mach_c = mach_n_c / (self.beta[_interval[0] - 1] - self.theta[_interval[0] - 1])

        nu_c = prandtl_meyer_fnc(mach_c) # mach ambiant après le choc
        delta_arr = self.theta[_interval[0] - 1] + self.theta[_interval[0]:_interval[1]]

        pressure_val = pressure_ratio(mach_c, _mach) * _pressure_up
        temperature_val = temperature_ratio(mach_c, _mach) * _temperature_up

        beta = self.beta[_interval[0]:_interval[1]]
        theta = self.theta[_interval[0]:_interval[1]]

        # vectors initialization
        pressure_arr, temperature_arr, density_arr, mach_n_arr, soundspeed_n_arr, velocity_n_arr, nu_arr, mach_arr = [np.zeros(len(self.profile.get_x())) for _ in range(8)]
        mach_i = 0
        for index in range(_interval[1] - _interval[0]):
            if index == 0:
                mach_i = mach_c

            nu_i = nu_c + delta_arr[index]
            mach_i = newton(inv_prandtl_meyer, mach_i, args=(nu_i, ))

            nu_arr[index] = nu_i
            mach_arr[index] = mach_i
            mach_n_arr[index] = mach_i * np.abs(np.sin(beta[index] - theta[index]))
            pressure_arr[index] = pressure_ratio(mach_c, mach_n_arr[index]) * pressure_val
            temperature_arr[index] = temperature_ratio(mach_n_arr[index], mach_c) *  temperature_val
            density_arr[index] = rho(pressure_arr[index], temperature_arr[index])
            soundspeed_n_arr[index] = np.sqrt(gamma * self.physic.atm.rs * temperature_arr[index])
            velocity_n_arr[index] = mach_n_arr[index] * soundspeed_n_arr[index]

        return {
            'pressure': pressure_arr,
            'temperature': temperature_arr,
            'density': density_arr,
            'mach_n': mach_n_arr,
            'mach_amb': mach_arr,
            'soundspeed_n': soundspeed_n_arr,
            'velocity_n': velocity_n_arr
        }

    def get_flow_characteristics(self):
        # vectors initialization
        pressure_arr, temperature_arr, density_arr, mach_n_arr, mach_amb_arr, soundspeed_n_arr, velocity_n_arr = [np.zeros(len(self.profile.get_x())) for _ in range(7)]

        for section_name in self.profile.get_section().keys():
            value = self.section_method[section_name]['interval']
            method = self.section_method[section_name]['method']

            if method == "Rankine-Hugoniot" or method == "Diedre":
                dictionary = self.variable_downstream_of_shock(_mach=self.mach_inf * np.sin(self.beta[value[0]:value[1]]))

                pressure_arr[value[0]:value[1]] = dictionary['pressure']
                temperature_arr[value[0]:value[1]] = dictionary['temperature']
                density_arr[value[0]:value[1]] = dictionary['density']
                mach_n_arr[value[0]:value[1]] = dictionary['mach_n']
                soundspeed_n_arr[value[0]:value[1]] = dictionary['soundspeed_n']
                velocity_n_arr[value[0]:value[1]] = dictionary['velocity_n']

                mach_amb_arr[value[0]:value[1]] = mach_n_arr[value[0]:value[1]] / np.sin(self.beta[value[0]:value[1]] - self.theta[value[0]:value[1]])

            elif method == "Prandtl-Meyer":
                dictionary = self.prandtl_meyer_variable(self.mach_inf * np.sin(self.beta[value[0] - 1]), value, self.physic.atm.pressure, self.physic.atm.temperature)
                pressure_arr[value[0]:value[1]] = dictionary['pressure']
                temperature_arr[value[0]:value[1]] = dictionary['temperature']
                density_arr[value[0]:value[1]] = dictionary['density']
                mach_n_arr[value[0]:value[1]] = dictionary['mach_n']
                soundspeed_n_arr[value[0]:value[1]] = dictionary['soundspeed_n']
                velocity_n_arr[value[0]:value[1]] = dictionary['velocity_n']
                mach_amb_arr[value[0]:value[1]] = dictionary['mach_amb']

        flow_characteristics = {
            'pressure': pressure_arr,
            'temperature': temperature_arr,
            'density': density_arr,
            'mach_n': mach_n_arr,
            'mach_amb': mach_amb_arr,
            'soundspeed_n': soundspeed_n_arr,
            'velocity_n': velocity_n_arr
        }
        return flow_characteristics

    def stagnation(self):
        x_val = self.profile.get_x()

        def var(_mach, _dictionary):
            gamma = self.physic.atm.gamma
            p_stag = (1 + 0.5 * (gamma - 1) * _mach ** 2) ** (gamma / (gamma - 1)) * _dictionary['pressure']
            t_stag = (1 + 0.5 * (gamma - 1) * _mach ** 2) * _dictionary['temperature']
            rho_stag = ((1 + 0.5 * (gamma - 1) * _mach ** 2) ** (1 / (gamma - 1))) * _dictionary['density']

            return {
                'p_stag': p_stag,
                't_stag': t_stag,
                'rho_stag': rho_stag
            }

        upstream_var_stag = var(_mach=np.full(len(x_val), self.mach_inf), _dictionary=self.physic.atm.get_atm_data())
        downstream_var_stag = var(_mach=self.flow_characteristics['mach_n'], _dictionary=self.flow_characteristics)

        return upstream_var_stag, downstream_var_stag

    def coefficient(self):
        def aero_forces(_coeff, _s_ref):
            return 0.5 * self.physic.atm.density * self.physic.velocity_x ** 2 * _s_ref * _coeff

        cp_arr, dsx_arr, dsy_arr, dcx_arr, dcy_arr = [np.zeros(len(self.theta)) for _ in range(5)]
        s_ref_arr = []

        gamma = self.physic.atm.gamma
        for section_name, section_value in self.profile.get_section().items():
            interval = self.section_method[section_name]['interval']

            if 'radius' in section_value:
                # pressure coefficient
                kp_star = (2 / gamma) * ((gamma + 1) / 2) ** (gamma / (gamma - 1)) * ((gamma + 1) / (2 * gamma)) ** (1 / (gamma - 1))
                cp_blunt = kp_star * np.sin(self.theta[interval[0]:interval[1]]) ** 2
                cp_arr[interval[0]:interval[1]] = cp_blunt

                # drag coefficient
                s_ref = np.abs(simpson(section_value['y'], x=section_value['x'])) # TODO: Not the same function as Pierre
                s_ref_arr.append(s_ref)

                ds = np.sqrt(np.diff(section_value['x']) ** 2 + np.diff(section_value['y']) ** 2)
                ds = np.append(ds, ds[-1])

                dsx_arr[interval[0]:interval[1]] = ds * np.cos(self.theta[interval[0]:interval[1]])
                dsy_arr[interval[0]:interval[1]] = ds * np.sin(self.theta[interval[0]:interval[1]])

                dcx_arr[interval[0]:interval[1]] = -cp_blunt * dsx_arr[interval[0]:interval[1]] / s_ref
                dcy_arr[interval[0]:interval[1]] = np.abs(-cp_blunt * dsy_arr[interval[0]:interval[1]] * s_ref)

            elif self.section_method[section_name]['method'] == "Diedre":
                # pressure coefficient
                kp_star = 2 * (gamma + 1) * (gamma + 7) / (gamma + 3) ** 2
                cp_diedre = kp_star * np.sin(self.theta[interval[0]:interval[1]]) ** 2
                cp_arr[interval[0]:interval[1]] = cp_diedre

                # drag coefficient
                s_ref = np.abs(simpson(section_value['y'], x=section_value['x']))
                s_ref_arr.append(s_ref)

                ds = np.sqrt(np.diff(section_value['x']) ** 2 + np.diff(section_value['y']) ** 2)
                ds = np.append(ds, ds[-1])

                dsx_arr[interval[0]:interval[1]] = ds * np.cos(self.theta[interval[0]:interval[1]])
                dsy_arr[interval[0]:interval[1]] = ds * np.sin(self.theta[interval[0]:interval[1]])

                dcx_arr[interval[0]:interval[1]] = -cp_diedre * dsx_arr[interval[0]:interval[1]] / s_ref
                dcy_arr[interval[0]:interval[1]] = np.abs(-cp_diedre * dsy_arr[interval[0]:interval[1]] / s_ref)

            elif self.section_method[section_name]['method'] == "Prandtl-Meyer":
                # pressure coefficient
                cp = 2 / (2 * self.mach_inf ** 2) * (
                            self.flow_characteristics['pressure'][interval[0]:interval[1]] / self.physic.atm.pressure - 1)
                cp_arr[interval[0]:interval[1]] = cp

                # drag coefficient
                s_ref = np.abs(simpson(section_value['y'], x=section_value['x']))
                s_ref_arr.append(s_ref)

                ds = np.sqrt(np.diff(section_value['x']) ** 2 + np.diff(section_value['y']) ** 2)
                ds = np.append(ds, ds[-1])

                dsx_arr[interval[0]:interval[1]] = ds * np.cos(self.theta[interval[0]:interval[1]])
                dsy_arr[interval[0]:interval[1]] = ds * np.sin(self.theta[interval[0]:interval[1]])

                dcx_arr[interval[0]:interval[1]] = -cp * dsx_arr[interval[0]:interval[1]] / s_ref
                dcy_arr[interval[0]:interval[1]] = np.abs(-cp * dsy_arr[interval[0]:interval[1]] / s_ref)

            else:
                # pressure coefficient
                q_inf = 0.5 * self.physic.atm.density * self.physic.velocity_x ** 2
                cp = (self.flow_characteristics['pressure'][interval[0]:interval[1]] - self.physic.atm.pressure) / q_inf
                cp_arr[interval[0]:interval[1]] = cp

                # drag coefficient
                s_ref = np.abs(simpson(section_value['y'], x=section_value['x']))
                s_ref_arr.append(s_ref)

                ds = np.sqrt(np.diff(section_value['x']) ** 2 + np.diff(section_value['y']) ** 2)
                ds = np.append(ds, ds[-1])

                dsx_arr[interval[0]:interval[1]] = ds * np.cos(self.theta[interval[0]:interval[1]])
                dsy_arr[interval[0]:interval[1]] = ds * np.sin(self.theta[interval[0]:interval[1]])

                dcx_arr[interval[0]:interval[1]] = -cp * dsx_arr[interval[0]:interval[1]] / s_ref
                dcy_arr[interval[0]:interval[1]] = np.abs(-cp * dsy_arr[interval[0]:interval[1]] / s_ref)

        self.cx = np.sum(dcx_arr) / np.sum(s_ref_arr)
        self.cy = np.sum(dcy_arr) / np.sum(s_ref_arr)

        self.fx = 2 * aero_forces(_coeff=self.cx, _s_ref=np.sum(s_ref_arr))
        self.fy = aero_forces(_coeff=self.cy, _s_ref=np.sum(s_ref_arr))

        return cp_arr, dcx_arr, dcy_arr
