import json

class JSONFilter():
    def __init__(self, type):
        self.type = type
        if self.type == 'scalar':
            filename = "scalar_feedback.json"
        elif self.type == 'preference':
            filename = "preference_feedback.json"
        self.json_data = self._read_json(filename)

    def _read_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
        

    def data_filter(self, model_name):
        filtered_data = [entry for entry in self.json_data if entry['model'] == model_name]
        return filtered_data

