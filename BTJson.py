import json

# Convert the following dictionary into JSON format
def convert_dictionary_json():
    data = {"key1" : "value1", "key2" : "value2"}
    prettyPrintedJson  = json.dumps(data, indent=2, separators=(",", " = "))

# Exercise 2: Access the value of key2 from the following JSON
def access_key_json():
    sampleJson = {"id" : 1, "name" : "value2", "age" : 29}
    data = json.loads(sampleJson)
    print(data['id'])

# Exercise 3: PrettyPrint following JSON data
def prettyPrint_json():
    sampleJson = {"key1" : "value2", "key2" : "value2", "key3" : "value3"}
    prettyPrintedJson  = json.dumps(sampleJson, indent=2, separators=(",", " = "))
    print(prettyPrintedJson)

# Sort JSON keys in and write them into a file
def sort_json():
    sampleJson = {"id" : 1, "name" : "value2", "age" : 29}
    print("Started writing JSON data into a file")
    with open("sampleJson.json", "w") as write_file:
        json.dump(sampleJson, write_file, indent=4, sort_keys=True)
    print("Done writing JSON data into a file")



