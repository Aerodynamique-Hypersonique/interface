import json

# Base class to create classes in json format
class JsonObject:
    def __init__(self):
        pass

    def from_dict(self, _dict):
        _dict.pop('class', None)
        print(_dict)
        self.__dict__.update(_dict)

    def to_dict(self):
        return {'baseclass': 'json'}
    def to_json(self):
        return json.dumps(self.to_dict())
