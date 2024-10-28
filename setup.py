from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='cvlface',
    version='0.1',
    author='Minchul Kim',
    author_email='kimminc2@msu.edu',
    description='CVL Holistic Face Library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mk-minchul/cvlface',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
)