# Documentation Generator

We use pydoc-markdown to generate the documentation. 
This code is for generating the documentation for the CVLface repository.
For those who are simply viewing documentations, do not worry about this code.

## Installation
For installing pydoc-markdown in ubuntu, run the following command:
```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install pydoc-markdown
```

## Docs Generation
For generating docs, run
```bash
cd <REPO_ROOT>
python docs/generator/generate.py
```

## Docs Serving
For serving
```bash
cd <REPO_ROOT>
docsify serve docs
```
