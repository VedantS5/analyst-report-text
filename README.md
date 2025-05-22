# Analyst Report Text

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Extract author metadata from text files derived from PDF analyst reports using LLMs with intelligent text chunking. This tool works in tandem with [pdf-md-toolkit](https://github.com/VedantS5/pdf-md-toolkit) for complete document processing.

## Overview

This specialized tool extracts author information from financial analyst reports using text-based processing. It takes text or markdown files (converted from PDFs) as input and applies LLM analysis to identify authors, titles, and contact information.

### Complete Workflow

1. **PDF to Text Conversion**: Use the [pdf-md-toolkit](https://github.com/VedantS5/pdf-md-toolkit) to convert PDF files to text/markdown format
2. **Text Analysis**: Process the converted files with this tool to extract author information

The system consists of three main components:

1. **02_markdown.py**: The main Python script that processes text files and uses Ollama to identify authors
2. **ollama_server_deployment.sh**: Shell script to deploy Ollama instances across available GPUs
3. **config.json**: Configuration file for customizing the behavior of the extraction tool

## Installation

```bash
# Clone the repository
git clone https://github.com/VedantS5/analyst-report-text.git
cd analyst-report-text

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

The repository contains these main components:

```
02_markdown.py               # Main script for text processing
ollama_server_deployment.sh  # Script to deploy Ollama instances
config.json                  # Configuration file
requirements.txt            # Required Python packages
setup.py                    # Package installation script
LICENSE                     # MIT License
```

> **Note:** The original development paths (`/N/project/fads_ng/analyst_reports_project/...`) referenced in code comments are specific to Indiana University's Quartz computing environment and should be adapted to your local environment.

## Purpose

This tool is designed to solve the problem of extracting structured author metadata from a large corpus of text documents. It specifically targets:

- Author names (requiring at least two words with proper capitalization)
- Professional titles (when available)
- Email addresses (when available)

The script is optimized for processing large document collections efficiently, skipping previously processed files and handling large documents by splitting them into manageable chunks.

## Setup and Environment

### System Requirements

This tool is designed to work with GPU acceleration for optimal performance:

- **Recommended**: NVIDIA GPUs with CUDA support
- Python 3.8+
- [Ollama](https://github.com/jmorganca/ollama) for running LLMs locally
- At least 16GB RAM (32GB+ recommended for large text files)
- [pdf-md-toolkit](https://github.com/VedantS5/pdf-md-toolkit) for PDF to text conversion

### For HPC/Cluster Users (Indiana University-specific)

> **Note:** The following commands are specific to Indiana University's computing resources. If you're using a different system, refer to your system's documentation for equivalent commands.

#### V100 Quartz:
```bash
srun -p gpu-debug --cpus-per-task 20 --gpus 4 --mem 40GB -A r01352 --time 1:00:00 --pty bash
```

#### A100 Big Red 200:
```bash
srun -p gpu-debug --cpus-per-task 30 --gpus 4 --mem 60GB -A r01352 --time 1:00:00 --pty bash
```

#### H100 Quartz Hopper:
```bash
srun -p hopper --cpus-per-task 40 --gpus 4 --mem 120GB -A r01352 --time 1:00:00 --pty bash
```

### General Setup (All Systems)

1. Ensure you have Ollama installed
   ```bash
   # For Linux/macOS
   curl -fsSL https://ollama.com/install.sh | sh
   
   # For other systems, see: https://github.com/jmorganca/ollama
   ```

2. Clone this repository and install dependencies:
   ```bash
   git clone https://github.com/VedantS5/analyst-report-text.git
   cd analyst-report-text
   pip install -r requirements.txt
   ```

3. Make the deployment script executable:
   ```bash
   chmod +x ollama_server_deployment.sh
   ```

4. Convert your PDF files to text/markdown using pdf-md-toolkit:
   ```bash
   # Install pdf-md-toolkit if not already installed
   pip install git+https://github.com/VedantS5/pdf-md-toolkit.git
   
   # Convert PDFs to text files
   pdf-md-toolkit --converter pymu --input /path/to/pdfs --output /path/to/text_files
   ```

Once the Ollama servers are running, you can proceed to execute the extraction script.

## Requirements

- Python 3.8+
- Ollama server(s) running (automatically detected)
- Required Python packages:
  - ollama (Python client library)
  - tiktoken (for token counting)
  - Standard libraries: os, csv, json, re, logging, threading, queue, time, argparse, subprocess, socket

## Configuration

The system uses a JSON-based configuration system, similar to the image-based module. All settings are stored in a `config.json` file in the project directory with the following structure:

```json
{
    "ollama": {
        "fallback_api_url": "http://localhost:11434/api/generate",
        "model": "gemma3:27b",
        "timeout": 180,
        "auto_detect": true,
        "port_range": [11434, 11465]
    },
    "processing": {
        "chunk_size": 6000,
        "chunk_overlap": 1000,
        "max_tokens": 8000
    },
    "input": {
        "directory": "/N/project/fads_ng/analyst_reports_project/data/analyst_reports_txt_page1/"
    },
    "output": {
        "directory": "/N/project/fads_ng/analyst_reports_project/data/csv_reports/",
        "csv_filename": "temp_author_report.csv"
    },
    "execution": {
        "max_files": 1000,
        "timeout_seconds": 120,
        "max_retries": 3
    },
    "prompt": {
        "template": "Extract information about the authors from the following text content..."
    },
    "debug": {
        "enabled": false,
        "log_level": "INFO"
    }
}
```

## Usage

### Basic Command

```bash
python /N/project/fads_ng/analyst_reports_project/codes/02/markdown_based/02_markdown.py
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to a custom config JSON file | `config.json` in script directory |
| `--output_csv` | Name of output CSV file (overrides config setting) | Value from config.json |
| `--max_files` | Maximum number of files to process (overrides config setting) | Value from config.json |
| `--chunk_size` | Target token count per chunk (overrides config setting) | Value from config.json |
| `--chunk_overlap` | Overlap between chunks (overrides config setting) | Value from config.json |
| `--debug` | Enable debug logging | Value from config.json |

### Examples

Process up to 500 files and save results to "author_analysis.csv":
```bash
python /N/project/fads_ng/analyst_reports_project/codes/02/markdown_based/02_markdown.py --output_csv author_analysis.csv --max_files 500
```

Process files with custom chunking parameters:
```bash
python /N/project/fads_ng/analyst_reports_project/codes/02/markdown_based/02_markdown.py --chunk_size 4000 --chunk_overlap 800
```

## Key Features

### Intelligent Author Extraction
- Detects author names, titles, and email addresses using AI analysis
- Handles multiple authors per document
- Verifies names have proper capitalization and at least two words
- Filters out specific patterns (e.g., "@mergent.com" emails)

### Input Requirements
- Only processes files with `.txt` extension
- Expects text files in the input directory specified in the configuration
- Does not process other file types (e.g., `.md`, `.docx`, etc.)

### Smart File Processing
- Automatically skips previously processed files
- Creates backups of existing output files
- Processes only the first page of PDF documents (from the source directory)
- Writes results in batches to avoid memory issues with large datasets

### Performance Optimization
- Multi-threaded processing using multiple Ollama servers
- Automatic detection of running Ollama servers
- Server cooldown mechanism when errors occur
- Timeout handling for problematic files
- Retry mechanism for failed processing attempts

### Document Handling
- Chunking of large documents to fit within token limits
- Smart overlap between chunks to maintain context
- Aggregation of author results from multiple chunks, removing duplicates

## How It Works

1. **Setup**: The script first checks for running Ollama servers and sets up logging.
   - Uses port scanning to auto-detect Ollama instances running on ports 11434-11465
   - Validates that at least one server is active before proceeding

2. **File Discovery**: It scans the input directory (`/N/project/fads_ng/analyst_reports_project/data/analyst_reports_txt_page1/`) for text files, comparing against previously processed files.
   - Builds a list of files that haven't been processed yet
   - Respects the `max_files` limit specified via command line

3. **Processing Queue**: Creates a queue of files to be processed and distributes them across available Ollama servers.
   - Uses multithreading with a worker pool to parallelize processing
   - Implements a server cooldown mechanism to handle errors gracefully

4. **Text Analysis**:
   - For each file, the content is read and tokenized using the `tiktoken` library
   - If the content exceeds the token limit, it's split into chunks with overlap
   - Chunks are processed separately with an LLM prompt that instructs the model to extract author information
   - Uses the Ollama client library to communicate with the Ollama API

5. **Results Collection**:
   - Author information from different chunks is aggregated using the `aggregate_author_results` function
   - Duplicate authors (based on name) are combined, keeping the most complete information
   - Implements retry logic for files that fail to process

6. **Output**: Results are written to a CSV file with columns for the filename and multiple authors (each with name, title, and email fields).
   - Maintains a backup of existing CSV files before modification
   - Dynamically adjusts column count based on the maximum number of authors found

## Output Format

The output CSV includes these columns:
- `filename`: The processed file name (cleaned from page markers)
- For each potential author (up to the maximum found):
  - `author_1_name`, `author_1_title`, `author_1_email`
  - `author_2_name`, `author_2_title`, `author_2_email`
  - etc.

## Error Handling

- Problematic files that time out are retried up to 3 times
- Detailed logging of processing errors
- Server cooldown when errors occur to prevent cascading failures
- Backup creation of existing CSV files before modifying them

## Environment Setup

### GPU Configuration

The deployment script can be configured for different GPU setups:

- `image`: Optimized for image processing (for analyst-report-vision)
- `h100`: Optimized for H100 GPUs - 8 instances per GPU
- `a100`: Optimized for A100 GPUs - 4 instances per GPU
- `v100`: Optimized for V100 GPUs - 3 instances per GPU
- `qwq`: Generic configuration for other GPU types - 2 instances per GPU

### Local Environment Configuration

For local installations, you may need to modify these settings:

1. Edit `ollama_server_deployment.sh` to match your GPU configuration
2. Update the `config.json` file with appropriate paths for your system
3. If running without multiple GPUs, you can use a simpler setup with a single Ollama instance

### Using Your Own LLM Models

The system is configured to use standard LLM models. If you want to use a different model:

1. Modify the model name in `config.json`
2. Update the `ollama_server_deployment.sh` script to pull your preferred model

## Integrated Workflow with pdf-md-toolkit

For a complete author extraction pipeline:

1. **Convert PDFs to text/markdown**:
   ```bash
   pdf-md-toolkit --converter pymu --input /path/to/pdfs --output /path/to/text_files
   ```

2. **Extract author information**:
   ```bash
   python 02_markdown.py --input_dir /path/to/text_files --output_csv author_results.csv
   ```

This two-step process separates concerns and allows for more flexibility in handling different document types and formats.

When running the script, you only need to specify the output CSV filename (not the full path) as the script automatically prepends the output directory path.