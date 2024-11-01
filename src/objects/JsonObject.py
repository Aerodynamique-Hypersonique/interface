import json

# Base class to create classes in json format
class JsonObject:
    def __init__(self):
        pass

    # Should be overwritten to handle the class variables
    def from_dict(self, _dict):
        _dict.pop('class', None)
        self.__dict__.update(_dict)

    def to_dict(self):
        return {'baseclass': 'JsonObject'}


    def to_json(self):
        return json.dumps(self.to_dict())
