import openai
import csv
import json
import chardet
import time
from tqdm import tqdm


# Function to read a file containing the user's OpenAI API key
def open_file(filepath):
	with open(filepath, 'r', encoding='utf-8') as infile:
		return infile.read()


# Set the API key
openai.api_key = open_file('openaiapikey.txt')


# Set the file_path variable
input_file_path = 'input.csv'


# Set the file_path variable
output_file_path = 'output'


# Set the chunk size
chunk_size = 100

# Write a prompt to pass to the completions endpoint
prompt = 'Using text mining techniques, discover the key trends and common themes among the data in this dataset. Structure the results with a description of the findings, then in a 10 point (not numbered) bulleted list complete with the number observed in the trend in parenthesis at the end of each line. Example 1. Most of the people in the dataset are from China (7 of 31). Make sure to keep the list organized in a logical order.\n'


def process_csv(input_file_path, chunk_size):
    """
    This function reads the contents of a large input CSV file, splits the contents into
    equal pieces based on the specified chunk size, and converts each piece into a JSON object.
    
    Parameters:
        input_file_path (str): The path to the CSV file.
        chunk_size (int): The number of rows to include in each chunk of data.

    Returns:
        list: A list of JSON objects, where each object represents a chunk of the
        original CSV data. Each chunk has at most chunk_size rows of the original data.
    """
    
    # Open the CSV file, read its contents and store in data as a list of rows
    with open(input_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

    # Calculate the number of chunks required to split data with chunk_size rows in each chunk
    num_chunks = (len(data) + chunk_size - 1) // chunk_size

    # Split data into chunks of chunk_size rows each
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Convert each chunk into a JSON string
    json_chunks = [json.dumps(chunk) for chunk in chunks] 

    return json_chunks


def analyze_text(json_chunks, prompt):
    """
    This function takes a list of JSON chunks, and submits each chunk to the OpenAI API
    for analysis. The results are returned as a list of strings. This version of the
    function implements exponential backoff manually to satisfy rate limit requirements
    of the Open AI API.

    Parameters:
        json_chunks (List[dict]): A list of JSON chunks.
        prompt (str): The prompt to use when calling the OpenAI API.

    Returns:
        List[str]: A list of strings, one for each chunk.
    """

    results = []
    wait_time = 1  # Initial wait time in seconds

    for chunk in json_chunks:
        # Combine the prompt with the chunk to create the input for the OpenAI API
        input_text = prompt + chunk

        # Call the API
        while True:
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=input_text,
                    max_tokens=1000,
                    n=1,
                    stop=None,
                    temperature=0.7,
                )
            except openai.exceptions.RateLimitExceededError:
                # If rate limit exceeded, wait for a bit before trying again
                time.sleep(wait_time)
                wait_time *= 2  # Double the wait time for the next attempt
            else:
                # If rate limit not exceeded, reset the wait time to 1 second
                wait_time = 1
                # Extract the text from the API response and append it to the list of results
                results.append(response.choices[0].text)
                break

    return results


def write_text(text, filename):
    """
    Writes the given text to a file in HTML format.
    
    text: list of str - the text to be written to the file
    filename: str - the name of the file (without the .html extension)
    """
    # create a file with the given filename and '.html' extension
    file = open(filename + ".html", "w")
    
    # write the HTML header
    file.write("<html>\n<head>\n<title>" + filename + "</title>\n")
    file.write("<link rel='stylesheet' type='text/css' href='stylesheet.css'>\n")
    file.write("</head>\n<body>\n")
    
    # write the table header
    file.write("<table>\n")

    # write the text
    file.write("<tr><td>\n<h3>Key Trends and Common Themes:</h3>\n")
    file.write("<p>" + text[0].replace("\n", "<br>\n") + "</p>\n")

    for result in text[1:]:
        file.write("<tr><td>" + result.replace("\n", "<br>\n") + "</td></tr>\n")

    # write the table footer
    file.write("</table>\n")

    # write the HTML footer
    file.write("</body>\n</html>")
    
    # close the file
    file.close()

    
if __name__ == '__main__':
    # Process the CSV file into chunks
    json_chunks = process_csv(input_file_path, chunk_size)
    # Analyze the text of each chunk
    results = []
    for chunk in tqdm(json_chunks, total=len(json_chunks),desc="Analyzing text"):
        results.append(analyze_text(json_chunks, prompt))
    # Write the analyzed results to a CSV file
    for text in tqdm(results, total=len(results), desc="Writing text"):
        write_text(text, output_file_path)