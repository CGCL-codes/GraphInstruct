from setuptools import setup, find_packages

setup(
    name='hust-LLM_Graph_Tasks_Generation',
    version='0.0.0',
    description='Setting up a python package',
    author='Xiran Song',
    author_email='xiransong@hust.edu.cn',
    # url='',
    packages=find_packages(include=['GTG']),
    install_requires=[
        "networkx >= 3.2.1",
        "numpy >= 1.21.5",
        "pandas >= 2.1.3",
        "PyYAML >= 6.0",
        "tqdm >= 4.63.0",
    ],
    classifiers=["License :: OSI Approved :: MIT License"],
    # extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    # setup_requires=['pytest-runner', 'flake8'],
    # tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': ['my-command=exampleproject.example:main']
    # },
    # package_data={'exampleproject': ['data/schema.json']}
)