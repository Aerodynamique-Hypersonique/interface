import math
import numpy as np
from src.objects.JsonObject import JsonObject



def load_profile_from_dict(_dict):
    class_map = {
        'Profile': Profile,
        'Parabolic': Parabolic,
        'Conical': Conical,
        'Ariane4': Ariane4
    }
    profile_class = class_map.get(_dict['class'])
    if profile_class:
        profile = profile_class()
        profile.from_dict(_dict)
        return profile
    return None


class Profile(JsonObject):
    def __init__(self, _section=None):
        super().__init__()
        if _section is None:
            _section = {'body':{ 'x': np.array([]), 'y': np.array([])}}
        self.section = _section # Dictionaries of the section

    def get_x(self):
        return np.array(self.section['body']['x'])

    def get_y(self):
        return np.array(self.section['body']['y'])

    def get_section(self):
        section = self.section.copy()
        section.pop('body', None)
        return section

    def to_dict(self):
        section_serializable = dict()
        for key, section in self.section.items():
            section_serializable[key] = {}
            for subkey, value in section.items():
                try:
                    section_serializable[key][subkey] = value.tolist()  # Convert numpy arrays to lists
                except AttributeError:
                    section_serializable[key][subkey] = value
        return {
            'class': 'Profile',
            'section': section_serializable,
        }



class Parabolic(Profile):
    def __init__(self, _angle=1, _length=5, _k=0):
        super().__init__()
        # K is the parabola type 0 <= K <= 1
        self.x = np.array([])
        self.y = np.array([])

        self.angle  = _angle
        self.length = _length
        self.k = _k
        self.radius = 0

        self.calculate_profile()

    def calculate_profile(self):
        self.radius = math.tan(self.angle * math.pi / 180) * self.length
        self.x = np.linspace(0, self.length, 1000)
        self.y = self.radius * (2 * self.x / self.length - self.k * (self.x / self.length) ** 2) / (2 - self.k)

        section = {
            'body': { # Body used to plot
                'x': self.x,
                'y': self.y
            },
            'structure': { # Structure used to calculate
                'x': self.x,
                'y': self.y,
                'radius': self.radius
            }
        }
        self.section = section

    def from_dict(self, _dict):
        if _dict['class'] == 'Parabolic':
            self.angle = _dict['angle']
            self.length = _dict['length']
            self.k = _dict['k']
            self.calculate_profile()

    def to_dict(self):
        profile_dict = super().to_dict()
        profile_dict.update({
            'class': 'Parabolic',
            'angle': self.angle,
            'length': self.length,
            'k': self.k
        })
        return profile_dict


class Conical(Profile):
    def __init__(self, _angle=1, _nose_rad=1, _length=5):
        super().__init__()
        self.angle = _angle
        self.length = _length
        self.nose_rad = _nose_rad
        self.radius = 0

        # Main structure
        self.x_structure = np.array([])
        self.y_structure = np.array([])

        # Nose structure
        self.x_nose = np.array([])
        self.y_nose = np.array([])

        # Total structure
        self.x = np.array([])
        self.y = np.array([])

        self.calculate_profile()

    def calculate_profile(self):
        nb_points = 1000
        # Create the radius of the end
        self.radius = math.tan(self.angle * math.pi / 180) * self.length

        # Find the tangency between circle and the cone
        xt = (self.length ** 2) / self.radius * np.sqrt(self.nose_rad ** 2 / (self.radius ** 2 + self.length ** 2))
        yt = xt * self.radius / self.length

        # Center of the nose
        x0 = xt + np.sqrt(self.nose_rad ** 2 - yt ** 2)

        # Apex point
        xa = x0 - self.nose_rad

        if xt < self.length:
            self.x_structure = np.linspace(xt, self.length, nb_points)
            self.y_structure = self.x_structure * self.radius / self.length

        # Add the circle
        self.x_nose = np.linspace(xa, xt, nb_points)
        self.y_nose = np.sqrt(self.nose_rad ** 2 - (self.x_nose - x0) ** 2)

        self.x = np.concatenate([self.x_nose[:nb_points - 1], self.x_structure])
        self.y = np.concatenate([self.y_nose[:nb_points - 1], self.y_structure])

        self.x -= self.x[0]

        section = {
            'structure': {
                'x': self.x,
                'y': self.y,
                'radius': self.nose_rad
            },
            'body': {
                'x': self.x,
                'y': self.y
            }
        }

        self.section = section

    def to_dict(self):
        profile_dict = super().to_dict()
        profile_dict.update({
            'class': 'Conical',
            'angle': self.angle,
            'length': self.length,
            'nose_rad': self.nose_rad
        })
        return profile_dict


    def from_dict(self, _dict):
        if _dict['class'] == 'Conical':
            # Load the base attributes
            self.angle = _dict['angle']
            self.length = _dict['length']
            self.nose_rad = _dict['nose_rad']

            self.calculate_profile()


class Ariane4(Profile):
    def __init__(self):
        super().__init__()

        self.x_cover = np.array([])
        self.y_cover = np.array([])

        # Deuxième partie : Section droite
        self.x_line_1 = np.array([])
        self.y_line_1 = np.array([])

        # Toisième partie : pente
        self.x_slope = np.array([])
        self.y_slope = np.array([])

        # Quatrième partie : dièdre
        self.x_diedre = np.array([])
        self.y_diedre = np.array([])

        # Cinquième partie : Section droite 2
        self.x_line_2 = np.array([])
        self.y_line_2 = np.array([])

        # Sixième partie : Section booster
        length_booster = 30
        self.radius_booster = 2.5
        self.x_booster = np.array([])
        self.y_booster = np.array([])

        self.x = np.array([])
        self.y = np.array([])

        self.calculate_profile()

    def calculate_profile(self):
        nb_points = 1000

        # Première partie du profil : la coiffe
        self.radius_cover = 2
        length_cover = 30
        self.x_cover = np.linspace(0, length_cover, nb_points)
        self.y_cover = self.radius_cover * (1 + np.sqrt(1 - np.square(np.divide(self.x_cover - length_cover, length_cover)))) - self.radius_cover

        # Deuxième partie : Section droite
        self.x_line_1 = np.linspace(length_cover, length_cover + 2 * self.radius_cover, nb_points)
        self.y_line_1 = np.full(nb_points, self.radius_cover)

        # Toisième partie : pente
        length_slope_1 = length_cover + 2 * self.radius_cover
        self.x_slope = np.linspace(length_slope_1, length_slope_1 + 3 * self.radius_cover, nb_points)
        slope_ang_1 = (0.5 * self.radius_cover - self.radius_cover) /  (self.x_slope[-1] - self.x_slope[0])
        self.y_slope = self.radius_cover + slope_ang_1 * (self.x_slope - self.x_slope[0])

        # Quatrième partie : dièdre
        x_start_diedre = length_slope_1 + 3 * self.radius_cover
        length_diedre = 2 * self.radius_cover
        slope_ang_2 = (1.5 * self.radius_cover - 0.5 * self.radius_cover) / length_diedre
        self.x_diedre = np.linspace(x_start_diedre, x_start_diedre + length_diedre, nb_points)
        self.y_diedre = 0.5 * self.radius_cover + slope_ang_2 * (self.x_diedre - self.x_diedre[0])

        # Cinquième partie : Section droite 2
        x_start_line_2 = x_start_diedre + length_diedre
        self.x_line_2 = np.linspace(x_start_line_2, x_start_line_2 + 3 * self.radius_cover, nb_points)
        self.y_line_2 = np.full(nb_points, 1.5 * self.radius_cover)

        # Sixième partie : Section booster
        x_start_booster = x_start_line_2 + 3 * self.radius_cover
        length_booster = 30
        self.radius_booster = 2.5
        self.x_booster = np.linspace(x_start_booster, x_start_booster + length_booster, nb_points)
        self.y_booster = 1.5 * self.radius_cover + self.radius_booster * (1 + np.sqrt(1 - np.square((self.x_booster - (x_start_booster + length_booster)) / length_booster))) - self.radius_booster

        self.x = np.concatenate([self.x_cover[:nb_points - 1], self.x_line_1[:nb_points - 1], self.x_slope[:nb_points - 1], self.x_diedre[:nb_points - 1], self.x_line_2[:nb_points - 1], self.x_booster])
        self.y = np.concatenate([self.y_cover[:nb_points - 1], self.y_line_1[:nb_points - 1], self.y_slope[:nb_points - 1], self.y_diedre[:nb_points - 1], self.y_line_2[:nb_points - 1], self.y_booster])

        section = {
            'cover': {
                'x': self.x_cover[:nb_points - 1],
                'y': self.y_cover[:nb_points - 1],
                'radius': self.radius_cover
            },
            'line_1': {
                'x': self.x_line_1[:nb_points - 1],
                'y': self.y_line_1[:nb_points - 1],
            },
            'slope': {
                'x': self.x_slope[:nb_points - 1],
                'y': self.y_slope[:nb_points - 1],
            },
            'diedre': {
                'x': self.x_diedre[:nb_points - 1],
                'y': self.y_diedre[:nb_points - 1],
            },
            'line_2': {
                'x': self.x_line_2[:nb_points - 1],
                'y': self.y_line_2[:nb_points - 1],
            },
            'booster': {
                'x': self.x_booster,
                'y': self.y_booster,
                'radius': self.radius_booster
            },
            'body': {
                'x': self.x,
                'y': self.y,
            }
        }
        self.section = section

    def from_dict(self, _dict):
        if _dict['class'] == 'Ariane4':
            self.calculate_profile()


    def to_dict(self):
        profile_dict = super().to_dict()

        profile_dict.update({
            'class': 'Ariane4',
        })
        return profile_dict