from collections import defaultdict
import openai
import json
import os
import subprocess
import argparse
import tempfile
import re

class ViolationData:
  # Class to store violation data
    def __init__(self, calculate_average=True):
      self.calculate_average = calculate_average
      self.total_violations = 0
      self.average_violations_per_line = 0 if calculate_average else None
      self.average_violations_per_file = 0
      self.most_frequent_violation = {"frequency": -1, "violation_type": ""}
      self.violation_frequencies = defaultdict(int)
      self.top_violations = []
    # Function to update the violation data
    def update_violation_data(self, total_violations, number_of_lines_in_code, total_results):
      self.total_violations += total_violations
      self.average_violations_per_file = self.total_violations / total_results
      if self.calculate_average:
          self.average_violations_per_line += (total_violations / number_of_lines_in_code) / total_results
    # Function to update the most frequent violation
    def update_most_frequent_violation(self):
        for violation_type, count in self.violation_frequencies.items():
            if count > self.most_frequent_violation["frequency"]:
                self.most_frequent_violation = {"frequency": count, "violation_type": violation_type}
    # Function to update the top three violations
    def update_top_violations(self):
        # Sort the violation frequencies and pick the top three
        sorted_violations = sorted(self.violation_frequencies.items(), key=lambda item: item[1], reverse=True)
        self.top_violations = sorted_violations[:3]
    # Function to convert the data to a dictionary
    def to_dict(self):
        # Return the data as a dictionary
        return {
            "total_violations": self.total_violations,
            "average_violations_per_line": self.average_violations_per_line,
            "average_violations_per_file": self.average_violations_per_file,
            "violation_frequencies": dict(self.violation_frequencies),
            "most_frequent_violation": self.most_frequent_violation,
            "top_violations": self.top_violations
        }
# Class to store linter data
class LinterData:
    # Class to store linter data
    def __init__(self, linter_name):
        # Create a ViolationData object to store the overall data
        self.overall = ViolationData(calculate_average=linter_name.lower() != "black")
        # Create a dictionary to store the data for each tag
        self.by_tag = defaultdict(lambda: ViolationData(calculate_average=linter_name.lower() != "black"))
   # Function to convert the data to a dictionary
    def to_dict(self):
        return {
            "overall": self.overall.to_dict(),
            "by_tag": {tag: data.to_dict() for tag, data in self.by_tag.items()}
        }
# Extract JSON from a directory
def extract_json_from_directory(abs_directory_path):
  # Create a list to store the JSON objects
  json_objects= {}
  # Iterate through each file in the directory
  for filename in os.listdir(abs_directory_path):
    # Get the absolute path of the file
    file_path = os.path.join(abs_directory_path, filename)
    # Check if it's a file
    if os.path.isfile(file_path):
      # Open the file and extract the JSON object
      with open(file_path, 'r') as file:
        # Read the file
        data = file.read()
        # Convert the JSON object to a Python dictionary
        json_object = json.loads(data)
        json_objects[json_object["id"]] = json_object
        # Close the file
        file.close()
  # Return the list of JSON objects
  return json_objects

# Function to create 100 unique problem descriptions
def create_problem_descriptions(keywords, num_problems=100):
  # Create a list to store the problem descriptions
  problem_descriptions = []
  index = 0
  # Iterate through the number of problems
  while index < num_problems:
      # Create a list to store the keywords
      cur_keywords = []
      cur_keyword =  keywords[index % len(keywords)]
      cur_keywords.append(cur_keyword)
      
      # Create a problem difficulty
      problem_difficulty = "Easy"
      if 50 <= (index % 100) < 80:
        problem_difficulty = "Medium"
        cur_keywords.append(keywords[(index + 1) % len(keywords)])
      elif (index % 100) >= 80:
        problem_difficulty = "Hard"
        cur_keywords.append(keywords[(index + 2) % len(keywords)])
      tags = cur_keywords.copy()
      tags.append(problem_difficulty)
      # Create a problem description
      problem_description = {"id": f"problem_{index + 1}", "description": "", "tags": tags,
                              "keywords": cur_keywords, "difficulty": problem_difficulty,
                              }
      # Add the problem description to the list                     
      problem_descriptions.append(problem_description)
      index += 1
  return problem_descriptions

# Function to prompt ChatGPT to generate Python code
def get_gpt_responses_to_problem_descriptions(problem_descriptions, response_folder):
  # Create a directory to store the JSON objects
  abs_directory_path = os.path.join(os.getcwd(), response_folder)
  # Check if the directory exists
  if not os.path.exists(abs_directory_path):
    os.makedirs(abs_directory_path)
  # Create a dictionary to store the coding problems
  coding_problems = {}
  # Iterate through each problem description
  for idx, problem_description in enumerate(problem_descriptions):
    # Create a prompt for ChatGPT
    chatgpt_prompt = f"Act as a Python developer and create a Python program. Here are the specifications for the Python program. Use the following keyword(s) in the following list as a start to create an idea for a problem that you would like to solve using python: {','.join(problem_description['keywords'])}. In addition, only return the raw code for the Python program. To ensure that the Python program is valid, act as Command Line Interface (you do not need to execute code) and make sure that the program runs correctly. Generate code for a functional python program as described above."
   
     
    # Make an API request to ChatGPT
    response = openai.Completion.create(
      
      engine="text-davinci-003",  # or whatever the latest model is
      prompt=chatgpt_prompt,
      max_tokens=2000,
      n=1,
      stop=None,
      temperature=0.5
    ) 
    # Save the response to a file
    filename = f"{problem_description['id']}.json"
    full_path = os.path.join(abs_directory_path, filename)
    # Write the JSON object to a file
    problem_object = {"id": problem_description['id'], 
         "description": problem_description['description'], 
         "code": response.choices[0].text.strip(),
         "tags": problem_description['tags'],
         
         "keywords": problem_description["keywords"], 
         "difficulty": problem_description["difficulty"],
                               }
    coding_problems[problem_description['id']] = problem_object
    # Write the JSON object to a file
    with open(full_path, 'w') as file:
      json_object = json.dumps(problem_object, indent=4)
      file.write(json_object)
      file.close()  # Close the file
    idx += 1
  return coding_problems


# List of keywords to use for the problems
keywords =[
    'list', 'asyncio', 'bytearray', 'bytes', 'ChainMap', 'Comprehension', 'Concurrency', 
    'ContextManager', 'Coroutine', 'Counter', 'Decorator', 'DefaultDict', 'DependencyInjection', 
    'Deque', 'dict', 'frozenset', 'GarbageCollection', 'Generator', 'Global state', 
    'heapq', 'IdiomaticPython', 'Introspection', 'Keyword expression', 'Lambda',  
    'Memoization', 'MemoryView', 'Metaclass', 'MonkeyPatching', 'NamedTuple', 'Non-default argument', 
    'OrderedDict', 'Polymorphism', 'Recursion', 'Reflection', 'Serialization', 
    'set', 'str', 'tuple', 'TypeHinting', 'UnitTesting', 'VirtualEnvironment', 'add', 'append', 
    'capitalize', 'clear', 'collections', 'copy', 'count', 'datetime', 'difference', 'discard', 
    'django', 'endswith', 'extend', 'find', 'flask', 'format', 'fromkeys', 'get', 'index', 'insert', 
    'intersection', 'issubset', 'issuperset', 'items', 'join', 'json', 'keys', 'lower', 'math', 
    'matplotlib', 'multiprocessing', 'numpy', 'os', 'pandas', 'pop', 'popitem', 'pytorch', 'random', 
    're', 'remove', 'replace', 'requests', 'reverse', 'scipy', 'setdefault', 'sklearn', 'socket', 
    'sort', 'split', 'startswith', 'strip', 'subprocess', 'sys', 
    'tensorflow', 'threading', 'union', 'upper',
]

# Function to save the code to a temporary file
def save_code_to_temp_file(code):
  # Create a temporary file
  with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp:
    tmp.write(code.encode('utf-8'))
    return tmp.name
      
# Function to run a linting tool and capture its output
def run_linting_tool(tool_name, file_name, *args):
  # Create a command to run the linting tool
  command = [tool_name, file_name] + list(args)
  result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  return result.stdout, result.stderr

# Function to parse and count the violations for flake8 and pylint
def parse_violations(lint_type, output):
  # Create a dictionary to store the violations
  violations = {"violations": defaultdict(int), "total_violations": 0, "most_frequent_violation": {"violation_type": "", "frequency": 0}}
  most_frequent_violation = ""
  most_frequent_violation_count = 0
  # Iterate through each line of the output
  for line in output.strip().split('\n'):
    # Check if the line is not empty
      if line:
        # Check if the line is a proper flake8 violation
        if lint_type == 'flake8' and not (line.split(':')[-1].startswith('{') or len(line.split(':')[-1].strip()) <= 1):
          # Get the violation type
          violation_type = line.split(':')[-1].strip()
          # Update the violation count
          violations["violations"][violation_type] += 1
          # Update the most frequent violation
          most_frequent_violation = most_frequent_violation if most_frequent_violation_count > violations["violations"][violation_type] else violation_type
          # Update the most frequent violation count
          most_frequent_violation_count = most_frequent_violation_count if most_frequent_violation_count > violations["violations"][violation_type] else violations["violations"][violation_type]
          # Update the total number of violations
          violations["total_violations"] += 1
        # Check if the line is a proper pylint violation
        elif lint_type == 'pylint':
          # Get the violation type
          violation_type = line.split(':')[-1].strip()
          # Check if the line is pylint violation
          if not (violation_type.startswith('*') or violation_type.startswith('Your code has been rated at') or violation_type.startswith('-')):
            # Update the violation count
            violations["violations"][violation_type] += 1
            # Update the most frequent violation
            most_frequent_violation = most_frequent_violation if most_frequent_violation_count > violations["violations"][violation_type] else violation_type
            # Update the most frequent violation count
            most_frequent_violation_count = most_frequent_violation_count if most_frequent_violation_count > violations["violations"][violation_type] else violations["violations"][violation_type]
            # Update the total number of violations
            violations["total_violations"] += 1
        
  # Update the most frequent violation
  violations["most_frequent_violation"]["violation_type"] = most_frequent_violation
  # Update the most frequent violation count
  violations["most_frequent_violation"]["frequency"] = most_frequent_violation_count
  return violations

# Main function to run all linting tools
def run_all_linters(code):
  # Save the code to a temporary file
  tmp_filename = save_code_to_temp_file(code)
  # Create a dictionary to store the results
  results = {}

  # flake8
  flake8_output, _ = run_linting_tool('flake8', tmp_filename)
  # Parse the flake8 output
  results['flake8'] = parse_violations('flake8', flake8_output)
  
  # pylint
  pylint_output, _ = run_linting_tool('pylint', '--output-format=text', tmp_filename)
  # Parse the pylint output
  results['pylint'] = parse_violations('pylint', pylint_output)

  # black
  _, black_output= run_linting_tool('black', '--check', tmp_filename)
  # Parse the black output
  results['black'] = {"violations": {} if 'would reformat' not in black_output else {"Non-compliant": 1} ,
                      "total_violations": 1 if 'would reformat' in black_output else 0,
                      "most_frequent_violation": {"violation_type": "would reformat" ,
                      "frequency": 1 } if 'would reformat' in black_output else {}}
  # remove the temporary file
  os.remove(tmp_filename)

  return results
# Function to count the number of non-empty lines in a string
def count_non_empty_lines(code_str):
    # Split the string into lines
    lines = code_str.splitlines()
    # Filter out the empty lines
    non_empty_lines = [line for line in lines if line.strip()]
    return len(non_empty_lines)



# Set up argument parsing
parser = argparse.ArgumentParser(description='Get ChatGPT responses and write to files.')
parser.add_argument('--generate_responses', action='store_true',
                    help='Force regeneration of ChatGPT responses')
args = parser.parse_args()

# Create problem descriptions
problem_descriptions = create_problem_descriptions(keywords)


response_folder = 'gpt_responses'

# Check if the folder exists
folder_exists = os.path.exists(response_folder)
# Prompt Chat GPT to generate the Python Interpreter output
abs_directory_path = os.path.join(os.getcwd(), response_folder)
# If the path already exists, then there is no need to reprompt ChatGPT
prompt_solutions = {}
# Check if the directory exists
if not os.path.exists(abs_directory_path) or args.generate_responses:
  # Prompt Chat GPT to generate the Python Interpreter output
    if not os.path.exists(abs_directory_path):
      os.makedirs(abs_directory_path)
    prompt_solutions = get_gpt_responses_to_problem_descriptions(problem_descriptions, response_folder)
else:
    # Extract the prompt solutions from the directory
    prompt_solutions = extract_json_from_directory(abs_directory_path)
    
# Create a directory to store the linting results
linting_results_directory_path = os.path.join(os.getcwd(), 'linter_results')
# Check if the directory exists
if not os.path.exists(linting_results_directory_path):
  os.makedirs
  
# Create a dictionary to store the results
linter_problem_results = {}
# Iterate through each prompt solution
for id, prompt_solution in prompt_solutions.items():

  # Run linters on the code string
  lint_results = run_all_linters(prompt_solution['code'])
  # Add the prompt solution to the dictionary
  linter_problem_results[id] = lint_results
# Create a list of linters
linter_names = ['flake8', 'pylint', 'black']  # Add other linters as needed
# Create a dictionary to store the overall results
overall_results = {linter: LinterData(linter) for linter in linter_names}
# Iterate through each prompt solution
for id, problem_results in linter_problem_results.items():
  # Save the results to a file
  filename = f"{id}.json"
  # Create a directory to store the JSON objects
  full_path = os.path.join(linting_results_directory_path, filename)
  # Write the prompt solution to a file
  with open(full_path, 'w') as file:
    # Convert the JSON object to a string
    json_object = json.dumps(
   problem_results, indent=4)
    # Write the string to the file
    file.write(json_object)
    # Close the file
    file.close()
  # Iterate through each linter
  for linter, problem_result in problem_results.items():
    # Obtain average number of violations for each linter
    number_of_lines_in_code = count_non_empty_lines(prompt_solutions[id]["code"])
    total_violations = problem_result["total_violations"]
    total_results = len(linter_problem_results)

    # Update overall data for each linter
    overall_results[linter].overall.update_violation_data(total_violations,  number_of_lines_in_code, total_results)
    # Update the violation frequencies
    for violation_type, violation_count in problem_result["violations"].items():
        overall_results[linter].overall.violation_frequencies[violation_type] += violation_count

    # Update data for each tag
    for tag in prompt_solutions[id]['tags']:
        overall_results[linter].by_tag[tag].update_violation_data(total_violations,number_of_lines_in_code, total_results)
        for violation_type, violation_count in problem_result["violations"].items():
            #print(violation_type)
            overall_results[linter].by_tag[tag].violation_frequencies[violation_type] += violation_count

    # Update most frequent violations
    overall_results[linter].overall.update_most_frequent_violation()
    # Update the top three violations
    overall_results[linter].overall.update_top_violations()
    for tag in prompt_solutions[id]['tags']:
        overall_results[linter].by_tag[tag].update_most_frequent_violation()
        overall_results[linter].by_tag[tag].update_top_violations()

      
  # Save the results to a file
  filename = "output.json"
  # Create a directory to store the JSON objects
  full_path = os.path.join(os.getcwd(), filename)
  # Convert overall_results to a dictionary format
  results_dict = {linter: data.to_dict() for linter, data in overall_results.items()}
  # Calculate the average number of violations per file across all linters. This is the output metric to integrate with cs8395/testing-suite
  results_dict["output"] = (results_dict["flake8"]["overall"]["average_violations_per_file"] + results_dict["pylint"]["overall"]["average_violations_per_file"] + results_dict["black"]["overall"]["average_violations_per_file"]) / 3
  # Write the prompt solution to a file
  with open(full_path, 'w') as file:
    # Convert the JSON object to a string
    json_object = json.dumps(
   results_dict, indent=4)
    # Write the string to the file
    file.write(json_object)
    # Close the file
    file.close()
        
