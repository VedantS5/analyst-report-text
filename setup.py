from setuptools import setup, find_packages

# Read the long description from README.md
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except:
    long_description = "Extract author metadata from PDF-derived text files using LLMs with intelligent text chunking"

setup(
    name="analyst-report-text",
    version="0.1.0",
    description="Extract author information from text/markdown files converted from PDFs, working in tandem with pdf-md-toolkit for complete document processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FADS Team",
    author_email="vedantshah@iu.edu",
    packages=find_packages(),
    py_modules=["02_markdown"],
    install_requires=[
        "ollama>=0.1.0",
        "tiktoken>=0.5.0",
        "pandas>=1.5.0",
        "tqdm>=4.64.0",
    ],
    entry_points={
        'console_scripts': [
            'analyst-report-text=02_markdown:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Topic :: Text Processing :: Markup :: Markdown",
    ],
    python_requires=">=3.8",
    project_urls={
        "Source": "https://github.com/VedantS5/analyst-report-text",
        "Issue Tracker": "https://github.com/VedantS5/analyst-report-text/issues",
        "Related Tool": "https://github.com/VedantS5/pdf-md-toolkit",
    }
)
