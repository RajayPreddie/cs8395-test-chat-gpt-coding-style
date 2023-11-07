from collections import defaultdict
import openai
import json
import os
import subprocess
import argparse
import tempfile


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
  problem_descriptions = []
  index = 0
  while index < num_problems:
      cur_keyword =  keywords[index % len(keywords)]
      is_well_written_code = True
      if index % 2 == 0:
        is_well_written_code = False
      well_written_code_tag = "well-written code" if is_well_written_code else "normal code"
      problem_difficulty = "Low"
      if 50 <= (index % 100) < 80:
        problem_difficulty = "Medium"
      
      elif (index % 100) >= 80:
        problem_difficulty = "High"
      # Create a problem description
      problem_description = {"id": f"problem_{index + 1}", "description": "", "tags": [cur_keyword, problem_difficulty, well_written_code_tag],
                              "keyword": cur_keyword, "difficulty": problem_difficulty,
                              "is_well_written_code": is_well_written_code
                              }
                            
      problem_descriptions.append(problem_description)
      index += 1
  return problem_descriptions


def create_coding_problems_from_problem_descriptions(problem_descriptions, response_folder):
  abs_directory_path = os.path.join(os.getcwd(), response_folder)
  if not os.path.exists(abs_directory_path):
    os.makedirs(abs_directory_path)
  coding_problems = {}
  for idx, problem_description in enumerate(problem_descriptions):
    well_written_code_prompt = "Another specification is to write well structured, readable, and maintainable code." if problem_description["is_well_written_code"] else ""
    chatgpt_prompt = f"Act as a Python developer and create a Python program. Here are the specifications for the Python program: use the following keyword for this Python program: {problem_description['keyword']}, the complexity of the code should be {problem_description['difficulty']}, only return the raw code for the Python program, and the code should be unique and not be a duplicate of any other code.{well_written_code_prompt} The code should be a valid Python program."
    print("New problem: ", idx + 1)
     
    # Make an API request to ChatGPT
    response = openai.Completion.create(
      
      engine="text-davinci-003",  # or whatever the latest model is
      prompt=chatgpt_prompt,
      max_tokens=60,
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
         
         "keyword": problem_description["keyword"], 
         "difficulty": problem_description["difficulty"],
         "is_well_written_code": problem_description["is_well_written_code"],
                               }
    coding_problems[problem_description['id']] = problem_object
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
    with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp:
        tmp.write(code.encode('utf-8'))
        return tmp.name
      
# Function to run a linting tool and capture its output
def run_linting_tool(tool_name, file_name, *args):
    command = [tool_name, file_name] + list(args)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout, result.stderr

# Function to parse and count the violations for flake8 and pylint
def parse_violations(lint_type, output):
    violations = defaultdict(int)
    for line in output.strip().split('\n'):
        if line:
            
            violation_type = line.split(':')[0].split(' ')[0]
            if lint_type == 'flake8' and not line.split(':')[-1].startswith('{'):
              violation_type = line.split(':')[-1]
              
              violations[violation_type] += 1
            elif lint_type == 'pylint':
              violation_type = line.split(':')[-1].strip()
              print("Pylint violation type: ", violation_type)
              violations[violation_type] += 1
    return violations

# Function to get cyclomatic complexity from Radon
def get_cyclomatic_complexity(file_name):
    output, _ = run_linting_tool('radon', 'cc', '-s', file_name)
    return output

# Function to get docstring style violations from pydocstyle
def get_docstring_violations(file_name):
    output, _ = run_linting_tool('pydocstyle', file_name)
    return parse_violations(output)

# Main function to run all linting tools
def run_all_linters(code):
    tmp_filename = save_code_to_temp_file(code)
    results = {}

    # flake8
    
    flake8_output, _ = run_linting_tool('flake8', tmp_filename)
    results['flake8'] = parse_violations('flake8', flake8_output)
  
    
    # pylint
    pylint_output, _ = run_linting_tool('pylint', '--output-format=text', tmp_filename)
    results['pylint'] = parse_violations('pylint', pylint_output)
    
    '''
    # mypy
    mypy_output, _ = run_linting_tool('mypy', tmp_filename)
    results['mypy'] = parse_violations(mypy_output)
    '''
    '''
    # black
    _, black_output= run_linting_tool('black', '--check', tmp_filename)
    
    results['black'] = 'Compliant' if 'would reformat' not in black_output else 'Non-compliant'
    '''
    
   
    # radon
    # results['radon'] = get_cyclomatic_complexity(tmp_filename)

    # Clean up the temporary file
    
    
    
    os.remove(tmp_filename)
  

    return results


# Set up argument parsing
parser = argparse.ArgumentParser(description='Get ChatGPT responses and write to files.')
parser.add_argument('--regenerate', action='store_true',
                    help='Force regeneration of ChatGPT responses')
args = parser.parse_args()


problem_descriptions = create_problem_descriptions(keywords)

### Prompt Generation Occurs Here ###
response_folder = 'gpt_responses'

# Check if the folder exists
folder_exists = os.path.exists(response_folder)
# Prompt Chat GPT to generate the Python Interpreter output
abs_directory_path = os.path.join(os.getcwd(), response_folder)
# If the path already exists, then there is no need to reprompt ChatGPT
prompt_solutions = {}
if not os.path.exists(abs_directory_path) or args.regenerate:
    if not os.path.exists(abs_directory_path):
      os.makedirs(abs_directory_path)
    prompt_solutions = create_coding_problems_from_problem_descriptions(problem_descriptions, response_folder)
else:
    # Extract the prompt solutions from the directory
    prompt_solutions = extract_json_from_directory(abs_directory_path)
    
# Save each prompt solution to a file in the coding_solutions directory
linting_results_directory_path = os.path.join(os.getcwd(), 'linter_results')
if not os.path.exists(linting_results_directory_path):
  os.makedirs(linting_results_directory_path)

linter_problem_results = {}
for id, prompt_solution in prompt_solutions.items():

  # Run linters on the code string
  lint_results = run_all_linters(prompt_solution['code'])
  print(lint_results)
  linter_problem_results[id] = lint_results
for linter, results in linter_problem_results.items():
  # Save the results to a file
  filename = f"{linter}.json"
  # Create a directory to store the JSON objects
  full_path = os.path.join(linting_results_directory_path, filename)
  # Write the prompt solution to a file
  with open(full_path, 'w') as file:
    # Convert the JSON object to a string
    json_object = json.dumps(
   results, indent=4)
    # Write the string to the file
    file.write(json_object)
    # Close the file
    file.close()
    for result in results:
      print(f"{linter}: {result}")
      if isinstance(result, dict):
        print(f"{linter} violations:")
        for violation_type, count in result.items():
            print(f"  {violation_type}: {count}")  
      else:
        print(f"{linter}: {result}")
  
  