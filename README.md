# Documentation for 02_markdown.py

## Overview

`02_markdown.py` is a specialized text processing tool that automatically extracts author information from document text files. The script uses AI-powered analysis through Ollama to identify authors along with their professional titles and email addresses from text documents.

Located at: `/N/project/fads_ng/analyst_reports_project/codes/02/markdown_based/02_markdown.py`

## Purpose

This tool is designed to solve the problem of extracting structured author metadata from a large corpus of text documents. It specifically targets:

- Author names (requiring at least two words with proper capitalization)
- Professional titles (when available)
- Email addresses (when available)

The script is optimized for processing large document collections efficiently, skipping previously processed files and handling large documents by splitting them into manageable chunks.

## Setup and Environment

### Requesting Interactive Computing Resources

Before running the code, you need to request appropriate computing resources based on your needs:

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

### Ollama Server Setup

After securing computing resources, you must deploy the Ollama servers before running the extraction script:

1. Navigate to the code directory:
   ```bash
   cd /N/project/fads_ng/analyst_reports_project/codes/02/
   ```

2. Launch Ollama servers using the deployment script:
   ```bash
   source ollama_server_deployment.sh [v100 or a100 or h100 or qwq]
   ```
   
   Example for Hopper GPUs:
   ```bash
   sh ollama_server_deployment.sh h100
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

## Production Environment

This script is designed to run in the specific project environment at:
`/N/project/fads_ng/analyst_reports_project/`

The script has hardcoded paths for:

- **Input files**: `/N/project/fads_ng/analyst_reports_project/data/analyst_reports_txt_page1/`
- **Output directory**: `/N/project/fads_ng/analyst_reports_project/data/csv_reports/`

When running the script, you only need to specify the output CSV filename (not the full path) as the script automatically prepends the output directory path.