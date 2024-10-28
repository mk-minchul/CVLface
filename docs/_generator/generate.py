import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["TODO.md"],
    pythonpath=True,
    dotenv=True,
)
import os, sys
sys.path.append(os.path.join(root))


import os
import re
from collections import defaultdict
from cvlface.general_utils.os_utils import get_all_files
import shutil


def reformat_python_code_blocks(markdown_content):
    # Regular expression to find blocks starting with >>>
    pattern = re.compile(r'(>>>(?:.|\n)*?)(?=\n\n|\n#|$)', re.MULTILINE)

    # Function to replace >>> with code block formatting
    def replace_with_code_block(match):
        block = match.group(0)
        # Remove >>> and strip leading and trailing whitespace
        formatted_block = block.replace('>>>', '').strip()
        # Encapsulate in triple backticks for Markdown code block
        return f'```python\n{formatted_block}\n```'

    # Replace all matches in the content
    reformatted_content = pattern.sub(replace_with_code_block, markdown_content)

    return reformatted_content


def merge_code_blocks(markdown_content):
    # Regular expression to find consecutive Python code blocks encapsulated in triple backticks with optional `python` language specifier
    pattern = re.compile(r'```python\n(.*?)\n```\n```python\n(.*?)\n```', re.DOTALL)

    # Function to merge found consecutive Python code blocks
    def merge_blocks(match):
        # Combine the content of the consecutive code blocks
        combined_block = f'{match.group(1)}\n{match.group(2)}'
        # Encapsulate in triple backticks with `python` language specifier for a single Markdown code block
        return f'```python\n{combined_block}\n```'

    # Keep applying the replacement until no more consecutive code blocks are found
    while re.search(pattern, markdown_content):
        markdown_content = re.sub(pattern, merge_blocks, markdown_content)

    return markdown_content


def find_python_files(root_dir):
    """
    Find all Python files at a specified subdirectory level relative to the root directory.

    Args:
    root_dir (str): The root directory to start searching from.
    level (int): The level of subdirectory to search in.

    Returns:
    dict: A dictionary with keys as directory paths and values as lists of Python files in those directories.
    """
    python_files = defaultdict(list)
    dirnames = os.listdir(root_dir)
    for dirname in dirnames:
        files = get_all_files(os.path.join(root_dir, dirname), extension_list=['.py'])
        for file in files:
            if file.endswith('.py'):
                python_files[os.path.join(root_dir, dirname)].append(file)
    return python_files

def extract_docstring(file_path, dir_path):
    """
    Extract docstrings that match the specified pattern from a Python file.

    Args:
    file_path (str): Path to the Python file.

    Returns:
    str: A docstring that is read from pydoc-markdown.
    """
    os.makedirs('./tmp', exist_ok=True)
    shutil.copy(file_path, './tmp/temp.py')
    if os.path.exists('/home/mckim/.local/bin/'):
        cmd = f'/home/mckim/.local/bin/pydoc-markdown pydoc-markdown.yml > tmp/temp.md'
    else:
        cmd = f'pydoc-markdown pydoc-markdown.yml > tmp/temp.md'
    os.system(cmd)
    with open('./tmp/temp.md', 'r', encoding='utf-8') as file:
        docstring = file.read()
    shutil.rmtree('./tmp')

    # trim the header
    header_str = '# temp\n\n'
    docstring = docstring[docstring.find(header_str)+len(header_str):]
    rel_file_path = os.path.relpath(file_path, dir_path)
    docstring = '# ' + rel_file_path + '\n\n' + docstring

    # format with special rules
    docstring = reformat_python_code_blocks(docstring)
    docstring = merge_code_blocks(docstring)
    docstring = docstring.replace('__init__.py', '\_\_init\_\_.py')
    if len(docstring.split('\n')) == 3:
        docstring = ''

    return docstring

def generate_markdown_files(python_files, output_dir):
    """
    Generate markdown files from the extracted docstrings organized by subdirectory level.

    Args:
    python_files (dict): A dictionary with keys as directory paths and values as lists of Python files in those directories.
    output_dir (str): The directory where to save the markdown files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dir_path, files in python_files.items():
        md_content = []
        for file in files:
            print(file)
            docstring = extract_docstring(file, dir_path)
            md_content.append(docstring)

        md_content = [content for content in md_content if content]

        if md_content:
            # Derive markdown file name from the directory path
            basename = os.path.basename(os.path.normpath(dir_path))
            dirname = os.path.basename(os.path.dirname(os.path.normpath(dir_path)))
            if 'run_' not in dirname:
                # subdivided
                md_filename = dirname + '_' + basename + '.md'
            else:
                md_filename = basename + '.md'
            content = '\n'.join(md_content)
            title = os.path.basename(dir_path).replace('_', ' ').title()
            divtitle = f'<div style="font-size: 36px; color: #333; font-weight: bold; text-align: center; margin-bottom: 20px;">{title}</div>'
            content = divtitle + '\n\n# \n' + content
            with open(os.path.join(output_dir, md_filename), 'w', encoding='utf-8') as md_file:
                md_file.write(content)

def subdivide(python_files, name='models'):
    model_key = [key for key in python_files.keys() if name in key][0]
    model_subdiv = {}
    for item in python_files[model_key]:
        rel_path = item.split(model_key+'/')[1]
        dirname = rel_path.split('/')[0]
        full_subdirname = os.path.join(model_key, dirname)
        if full_subdirname not in model_subdiv:
            model_subdiv[full_subdirname] = []
        model_subdiv[full_subdirname].append(item)
    del python_files[model_key]
    python_files.update(model_subdiv)

if __name__ == '__main__':

    # Example usage
    os.chdir(os.path.dirname(__file__))
    root_dir = '../../cvlface/research/recognition/code/run_v1'
    root_dir = os.path.abspath(root_dir)
    output_dir = '../../docs/generated'
    python_files = find_python_files(root_dir)

    # subdivide folders if needed
    subdivide(python_files, name='models')
    subdivide(python_files, name='aligners')

    generate_markdown_files(python_files, output_dir)