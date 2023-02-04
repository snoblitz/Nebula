import openai
import csv
import json
import chardet


# Function to read a file containing the user's OpenAI API key
def open_file(filepath):
	with open(filepath, 'r', encoding='utf-8') as infile:
		return infile.read()


# Set the API key
openai.api_key = open_file('openaiapikey.txt')


# Set the max_tokens variable
max_input_tokens = 3000


# Set the file_path variable
input_file_path = 'input.csv'


# Set the file_path variable
output_file_path = 'output.txt'


# Write a prompt to pass to the completions endpoint
prompt = 'Discover the key trends and common themes (limit your respone to less than 1000 tokens) among the data in this dataset by analyzing the following information:\n'


import chardet

def get_chunk_size(input_file_path, max_input_tokens):
    """
    This function takes an input file path and a maximum number of tokens, and returns an appropriate
    chunk size based on the number of tokens in the file.

    Parameters:
        input_file_path (str): The path to the CSV file.
        max_tokens (int): The maximum number of tokens per chunk.

    Returns:
        int: The chunk size, in number of rows.
    """

    avg_tokens_per_row = 0
    row_count = 0

    with open(input_file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    with open(input_file_path, 'r', encoding=encoding) as f:
        for row in csv.reader(f):
            avg_tokens_per_row += len(row)
            row_count += 1

    avg_tokens_per_row = avg_tokens_per_row / row_count

    chunk_size = max_input_tokens // avg_tokens_per_row
    print(chunk_size)

    return int(chunk_size)



def process_csv(input_file_path, chunk_size):
    """
    This function takes a large input CSV file and splits it into several equal pieces, 
    converting each piece into a JSON object. The function returns a list of 
    JSON objects, where each object represents a chunk of the original CSV data.

    Parameters:
        input_file_path (str): The path to the CSV file.
        chunk_size (int): The size of each chunk to split the data into.

    Returns:
        list: A list of JSON objects, where each object represents a chunk of the
        original CSV data.
    """
    
    # Open the CSV file and read its contents
    with open(input_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    # Calculate the number of chunks based on the specified chunk size
    num_chunks = len(data) // chunk_size + (len(data) % chunk_size != 0)

    # Split the data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Convert each chunk into a JSON object
    json_chunks = [json.dumps(chunk) for chunk in chunks]

    return json_chunks


def analyze_text(json_chunks, prompt):
    """
    This function takes a list of JSON chunks, and submits each chunk to the OpenAI API
    for analysis. The results are returned as a list of dictionaries.

    Parameters:
        json_chunks (List[dict]): A list of JSON chunks.
        prompt (str): The prompt to use when calling the OpenAI API.

    Returns:
        List[dict]: A list of dictionaries, one for each chunk.
    """

    results = []

    for chunk in json_chunks:
        # Combine the prompt with the chunk to create the input for the OpenAI API
        input_text = prompt + chunk

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=input_text,
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Extract the text from the API response and add it to the dictionary
        result = {"text": response.choices[0].text}

        # Append the result to the list of results
        results.append(result)

    return results



def write_text(results, output_file_path):
    """
    Writes the results list to a text file at the specified filepath.

    Parameters:
        results (List[dict]): A list of results in dictionary form.
        output_file_path (str): The filepath where the text file will be saved.

    Returns:
        None
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for result in results:
                outfile.write(str(result) + '\n')
        print(f"Text file written successfully to {output_file_path}")
    except Exception as e:
        print(f"Failed to write text file to {output_file_path}: {e}")



if __name__ == '__main__':
    # Get the size of the chunk to process the input file
    chunk_size = get_chunk_size(input_file_path, max_input_tokens)
    # Process the CSV file into chunks
    json_chunks = process_csv(input_file_path, chunk_size)
    # Analyze the text of each chunk
    results = analyze_text(json_chunks, prompt)
    # Write the analyzed results to a CSV file
    write_text(results, output_file_path)