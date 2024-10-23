import math
import numpy as np
import json

class Profile:
    def __init__(self, _x=np.array([]), _y=np.array([]), _angle=0, _length=0, _radius=0.0):
        self.x = _x
        self.y = _y
        self.angle = _angle
        self.length = _length
        self.radius = _radius

    def from_dict(self, _dict):
        self.x = np.array(_dict['x'])
        self.y = np.array(_dict['y'])
        self.angle = _dict['angle']
        self.length = _dict['length']
        self.radius = _dict['radius']

    def get_x(self):
        return self.x
    def get_y(self):
        return self.y



class Parabolic(Profile):
    def __init__(self, _angle=0, _length=0, _k=0):
        # K is the parabola type 0 <= K <= 1
        self.angle  = _angle
        self.length = _length
        self.radius = math.tan(self.angle * math.pi / 180) * _length
        self.k = _k

        self.x = np.linspace(0, _length, 1000)
        self.y = self.radius * (2 * self.x / self.length - self.k * (self.x / self.length) ** 2) / (2 - self.k)

        Profile.__init__(self, self.x, self.y, self.angle, self.length, self.radius)

    def from_dict(self, _dict):
        if _dict['name'] == 'Parabolic':
            Profile.from_dict(self, _dict['profile'])
            self.k = _dict['k']

    def to_dict(self):
        return {'name': 'Parabolic',
                'profile': {'x': self.get_x().tolist(),
                            'y': self.get_y().tolist(),
                            'angle': self.angle,
                            'length': self.length,
                            'radius': self.radius,
                            },
                'k': self.radius
                }

    def to_json(self):
        return json.dumps(self.to_dict())


class Conical(Profile):
    def __init__(self, _angle=0, _nose_rad=0, _length=0):
        self.angle      = _angle
        self.length     = _length
        self.nose_rad  = _nose_rad

        # Create the radius of the end
        self.radius = math.tan(self.angle * math.pi / 180) * self.length

        # Find the tangency between circle and the cone
        xt = (self.length ** 2) / self.radius * np.sqrt(self.nose_rad ** 2 / (self.radius ** 2 + self.length ** 2))
        yt = xt * self.radius / self.length

        # Center of the nose
        x0 = xt + np.sqrt(self.nose_rad ** 2 - yt ** 2)

        # Apex point
        xa = x0 - self.nose_rad

        # Main structure
        self.x = np.array([])
        self.y = np.array([])

        if xt < self.length:
            self.x = np.linspace(xt, self.length, 1000)
            self.y = self.x * self.radius / self.length

        # Add the circle
        self.x_nose = np.linspace(xa, xt, 500)
        self.y_nose = np.sqrt(self.nose_rad ** 2 - (self.x_nose - x0) ** 2)

        Profile.__init__(self, self.x, self.y, self.angle, self.length, self.radius)

    def from_dict(self, _dict):
        if _dict['name'] == 'Conical':
            Profile.from_dict(self, _dict['profile'])
            self.x_nose = np.array(_dict['x_nose'])
            self.y_nose = np.array(_dict['y_nose'])

    def get_x_nose(self):
        return self.x_nose

    def get_y_nose(self):
        return self.y_nose

    def to_dict(self):
        return {'name': 'Conical',
                'profile': {'x': self.get_x().tolist(),
                            'y': self.get_y().tolist(),
                            'angle': self.angle,
                            'length': self.length,
                            'radius': self.radius,
                            },
                'nose_rad': self.nose_rad,
                'x_nose': self.x_nose.tolist(),
                'y_nose': self.y_nose.tolist()
                }

    def to_json(self):
        return json.dumps(self.to_dict())

