import json
import jsonschema
from jsonschema import validate

# Describe what kind of json you expect.
studentSchema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "rollnumber": {"type": "number"},
        "marks": {"type": "number"},
    },
}
# Check validateJson
# def validateJSON(jsonData):
#     jsonData = """{"name": "jane doe", "salary": 9000, "email": "jane.doe@pynative.com"}"""
#     try:
#         json.loads(jsonData)
#     except ValueError as err:
#         return False
#     return True

def validateJson(jsonData):
    try:
        validate(instance=jsonData, schema=studentSchema)
    except jsonschema.exceptions.ValidationError as err:
        return False
    return True

# Convert json to python object.
jsonData = json.loads('{"name": "jane doe", "rollnumber": "25", "marks": 72}')
# validate it
isValid = validateJson(jsonData)
print("IS VALID", isValid)
if isValid:
    print(jsonData)
    print("Given JSON data is Valid")
else:
    
    print(jsonData)
    print("Given JSON data is InValid")
