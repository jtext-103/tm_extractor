from setuptools import setup, find_packages
# Ensure that reading long_description won't cause errors
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

setup(
    name="tm_extractor",  # The name of your package
    version="0.1",  # Version number
    entry_points={  # Entry point for command-line interface
        'console_scripts': [
            'tm_extractor = tm_extractor.run_tm_extractor:main_function',],
    },

    # # Include tm_extractor and its submodules
    packages=find_packages(),  # Include all packages
    package_data={  # Include additional files
            'tm_extractor': ['requirement.txt', 'README.md'],  # Include the requirement.txt and README.md files
            'tm_extractor.tearing_mode_extractor': ['default_config_json.json'], # Include the default_config_json.json file
# Include the default_config_json.json file
    },
    include_package_data=True,  # Include package data
    install_requires=[  # Libraries that the package depends on
        "jddb @ git+https://github.com/jtext-103/jddb.git@master#egg=jddb&subdirectory=code/JDDB",
    ],
    author="Luo Runyu",  # Author's name
    author_email="lryaurora@gmail.com",  # Author's email
    description="A Software Toolkit to Extract Tearing Modes Features",  # Short description of the package
    long_description=long_description,  # Long description from README file
    long_description_content_type="text/markdown",  # Specify the format of the README file
    classifiers=[  # Classifications to help users find your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # License type (modify according to your license)
        "Operating System :: OS Independent",  # This package works across different OS
    ],
    python_requires='>=3.7',  # Minimum required Python version
)

