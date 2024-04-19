import yaml
import base64
import json
import argparse
import sys

def encode_answer(answer):
    # Convert any non-string answer to string before encoding
    answer_str = str(answer)
    answer_bytes = answer_str.encode('utf-8')  # Convert to bytes
    encoded_bytes = base64.b64encode(answer_bytes)
    encoded_str = encoded_bytes.decode('utf-8')  # Convert bytes back to string
    return encoded_str

"""
def encode_data(answer):
    # Convert any non-string answer to string before encoding
    json_str = json.dumps(answer)
    encoded_bytes = base64.b64encode(json_str.encode('utf-8'))
    return encoded_bytes.decode('utf-8')
"""


def preprocess_yaml(input_file, output_file):
    print("input file: ", input_file)
    print("output file: ", output_file)
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

    if data is None:
        sys.exit(f"Error: Empty or incorrect {input_file}")
    
    for question in data.get('questions', []):
        parts = question.get('parts', [])
        for i, part in enumerate(parts):
            # print("part: ", part)
            # Encode all answers
            #part['answer'] = encode_answer(part['answer'])
            if 's_answers' in part and isinstance(part['s_answers'], list):
                #print(f"{part['s_answers']=}")
                answers = part['s_answers']
                for j, answ in enumerate(answers):
                    part['s_answers'][j] = encode_data(answ)
            if 'i_answers' in part and isinstance(part['i_answers'], list):
                #print(f"{part['i_answers']=}")
                answers = part['i_answers']
                for j, answ in enumerate(answers):
                    part['i_answers'][j] = encode_data(answ)
            if 's_answer' in part:
                part['s_answer'] = encode_data(part['s_answer'])
            if 'i_answer' in part:
                part['i_answer'] = encode_data(part['i_answer'])
            #print(f"{len(parts)=}, {i=}")
            parts[i] = part
            #print("part= ", part)
        question['parts'] = parts
    
    with open(output_file, 'w') as file:
        yaml.dump(data, file, sort_keys=False)

"""
def decode_answer(encoded_str):
    decoded_bytes = base64.b64decode(encoded_str)
    decoded_str = decoded_bytes.decode('utf-8')
    print(f"{type(decoded_str)=}")
    return decoded_str
"""

"""
def decode_data(encoded_str):
    # Decode from Base64 and then deserialize from JSON string to original data type
    decoded_bytes = base64.b64decode(encoded_str)
    json_str = decoded_bytes.decode('utf-8')
    print("==> json_str= ", json_str)
    try:
        # Attempt to load the JSON string
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If JSON decoding fails, return the string directly
        return json_str
    #return json.loads(json_str)
"""

"""
def encode_data(answer):
    # Annotate the data with its type information
    data_to_encode = {"type": "list" if isinstance(answer, list) else "set" if isinstance(answer, set) else "other",
                      "data": list(answer) if isinstance(answer, (set, list)) else answer}
    json_str = json.dumps(data_to_encode)
    encoded_bytes = base64.b64encode(json_str.encode('utf-8'))
    return encoded_bytes.decode('utf-8')


def decode_data(encoded_str):
    decoded_bytes = base64.b64decode(encoded_str)
    json_str = decoded_bytes.decode('utf-8')
    decoded_data = json.loads(json_str)
    print("decoded: ", decoded_data)
    
    # Reconstruct the original type based on the type information
    if decoded_data["type"] == "set":
        return set(decoded_data["data"])
    else:
        return decoded_data["data"]
"""

"""
    if decoded_data["type"] == "list":
        return decoded_data["data"]
    elif decoded_data["type"] == "set":
        return set(decoded_data["data"])
    return decoded_data["data"]
"""

# Example: Encoding with type hints
def encode_data(data):
    if isinstance(data, set):
        encoded = json.dumps({"type": "set", "data": list(data)})
    else:
        #print("data= ", data)
        encoded = json.dumps(data)  # Directly use JSON for other types
    return base64.b64encode(encoded.encode()).decode()

# Example: Decoding with type handling
def decode_data(encoded):
    decoded = json.loads(base64.b64decode(encoded).decode())
    # Not clear why the next two lines are needed
    if isinstance(decoded, dict) and decoded.get("type") == "set":
        return set(decoded["data"])
    if isinstance(decoded, str) and decoded[0:3] == 'set':
        return eval(decoded)
    return decoded



#----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script and what it does.")

    # Add the arguments
    parser.add_argument("-f", "--filename", type=str, help="The filename to process", required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Usage (do not include yaml extension)
    input_file = args.filename  + ".yaml"
    output_file = f"preprocessed_{input_file}"
    preprocess_yaml(input_file, output_file)
