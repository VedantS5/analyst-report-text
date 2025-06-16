import os
import csv
import json
import re
import logging
import threading
import queue
import time
import argparse
from typing import List, Dict, Set, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# Helper to read CSV layout (wide vs long) from configuration
# ---------------------------------------------------------------------------

def get_csv_layout() -> str:
    """Return desired CSV layout style. Defaults to 'wide'."""
    return get_config().get('output', {}).get('csv_layout', 'wide').lower()
from ollama import Client
import tiktoken
import subprocess
import socket
import sys

def load_json_config(config_file=None) -> Dict[str, Any]:
    """
    Load configuration from JSON file with fallback to default values if file not found.
    
    Args:
        config_file: Path to the JSON configuration file. If None, looks in the script directory.
        
    Returns:
        Dict containing configuration values.
    """
    # Default configuration as fallback
    default_config = {
        "ollama": {
            "fallback_api_url": "http://localhost:11434/api/generate",
            "model": "gemma3:27b",
            "timeout": 180,
            "auto_detect": True,
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
            "template": "Extract information about the authors from the following text content of a research report or financial document. Carefully look for names, titles, and email addresses.\n\nRules:\n1. Names must be properly capitalized and have at least two words (e.g., John Smith)\n2. Look for titles that often accompany author names (e.g., Senior Analyst, Chief Economist)\n3. Email addresses typically follow standard formatting (name@domain.com)\n4. Only include actual authors, not referenced people\n5. If multiple authors, list them all\n6. If email domain is mergent.com, exclude it as it's not an author\n\nRespond ONLY in valid JSON format:\n{\"authors\": [{\"name\": \"Full Name\", \"title\": \"Professional Title\", \"email\": \"email@address.com\"}]}\nIf you find no valid authors, respond with {\"authors\": []}\n\nTEXT CONTENT:\n{{TEXT_CONTENT}}"
        },
        "parsing": {
            "type": "json",                  # "json" (default) or "regex"
            "authors_key": "authors",        # Key containing the list of authors in JSON output
            "name_key": "name",             # Field name for author name
            "title_key": "title",           # Field name for author title
            "email_key": "email",           # Field name for author email
            "skip_domains": ["mergent.com"], # Email domains to ignore
            "regex_pattern": "",            # Full regex with named groups (if type == 'regex')
            "name_group": "name",           # Named group for name (regex mode)
            "title_group": "title",         # Named group for title (regex mode)
            "email_group": "email"          # Named group for email (regex mode)
        },
        "debug": {
            "enabled": False,
            "log_level": "INFO"
        }
    }
    
    # If no config file specified, check in the script directory
    if config_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, 'config.json')
    
    # Try to load from the config file
    try:
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            
        # Log successful loading of config file
        logging.info(f"Loaded configuration from {config_file}")
            
        # Merge with default config (keeps values from loaded_config when keys overlap)
        # This ensures any new config options added later will have defaults
        for section in default_config:
            if section in loaded_config:
                default_config[section].update(loaded_config[section])
            # If a whole section is missing in the loaded config, keep the default section
                
        # Update sections that exist in loaded config but not in default config
        for section in loaded_config:
            if section not in default_config:
                default_config[section] = loaded_config[section]
                
    except FileNotFoundError:
        logging.warning(f"Config file {config_file} not found. Using default configuration.")
    except json.JSONDecodeError:
        logging.error(f"Error parsing the config file {config_file}. Using default configuration.")
    except Exception as e:
        logging.error(f"Unexpected error loading config: {e}. Using default configuration.")
    
    # Set up logging based on configuration
    log_level_name = default_config.get('debug', {}).get('log_level', 'INFO')
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    return default_config

# Configuration handling
def get_config():
    """Get the current configuration, loading it if necessary"""
    if not hasattr(get_config, "_config"):
        get_config._config = load_json_config()
    return get_config._config

# Helper functions to get config values
def get_chunk_size():
    return get_config()['processing']['chunk_size']

def get_chunk_overlap():
    return get_config()['processing']['chunk_overlap']

def get_max_tokens():
    return get_config()['processing']['max_tokens']

def get_token_count(text: str) -> int:
    """
    Estimate the number of tokens in the given text.
    Uses tiktoken if available, falls back to simple word count estimate if not.
    """
    try:
        # Try to use tiktoken for accurate count
        encoding = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding
        return len(encoding.encode(text))
    except (ImportError, AttributeError):
        # Fallback method: approximate tokens by word count
        return len(text.split()) * 4 // 3  # Rough estimate: 1 token = 0.75 words

def chunk_text(text: str) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens with chunk_overlap overlap.
    Uses configuration values from get_config().
    """
    # Get chunk sizes from config
    chunk_size = get_chunk_size()
    chunk_overlap = get_chunk_overlap()
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
        
        paragraph_size = get_token_count(paragraph)
        
        # If a single paragraph is too large, we need to split it further
        if paragraph_size > chunk_size:
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence_size = get_token_count(sentence)
                if current_size + sentence_size <= chunk_size:
                    current_chunk.append(sentence)
                    current_size += sentence_size
                else:
                    # Current chunk is full, save it and start a new one
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_size = sentence_size
        else:
            # Normal paragraph handling
            if current_size + paragraph_size <= chunk_size:
                current_chunk.append(paragraph)
                current_size += paragraph_size
            else:
                # Current chunk is full, save it and start a new one
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                
                # Start new chunk with overlap
                # Find paragraphs to include from previous chunk for context
                overlap_size = 0
                overlap_paragraphs = []
                
                for p in reversed(current_chunk):
                    p_size = get_token_count(p)
                    if overlap_size + p_size <= chunk_overlap:
                        overlap_paragraphs.insert(0, p)
                        overlap_size += p_size
                    else:
                        break
                
                current_chunk = overlap_paragraphs + [paragraph]
                current_size = overlap_size + paragraph_size

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def aggregate_author_results(chunk_results: List[List[Dict]]) -> List[Dict]:
    """
    Combine author results from multiple chunks, removing duplicates.
    Authors are considered duplicates if they share the same name.
    When duplicates are found, we keep the most complete version of the author data.
    """
    authors_by_name = {}
    
    # Process all chunks
    for authors in chunk_results:
        for author in authors:
            name = author.get('name', '').strip()
            if not name:
                continue
                
            # If we haven't seen this author before, add them
            if name not in authors_by_name:
                authors_by_name[name] = author.copy()
            else:
                # If we've seen this author, merge data, preferring non-empty values
                existing = authors_by_name[name]
                for field in ['title', 'email']:
                    if not existing.get(field) and author.get(field):
                        existing[field] = author.get(field)
    
    # Convert back to list
    return list(authors_by_name.values())

def append_results_to_csv(output_csv: str, new_results: List[Dict], max_authors: int) -> None:
    """Append results to CSV, honouring configured layout."""
    layout = get_csv_layout()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Determine if these results contain authors or generic columns
    sample = new_results[0] if new_results else {}
    if 'authors' not in sample:
        # Generic (score) mode – just write simple columns
        headers = ['filename'] + [k for k in sample.keys() if k != 'filename']
        file_exists = os.path.exists(output_csv)
        try:
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                for row in new_results:
                    writer.writerow(row)
            logging.info(f"Appended {len(new_results)} score rows to {output_csv}")
        except Exception as e:
            logging.error(f"Error writing to CSV file {output_csv}: {e}")
        return

    # ------------------------------------------------------------------
    # Author extraction mode (existing logic)
    # ------------------------------------------------------------------
    if layout == 'long':
        headers = ['filename', 'author_name', 'author_title', 'author_email']
        file_exists = os.path.exists(output_csv)
        try:
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                for entry in new_results:
                    filename = entry.get('filename', '')
                    authors = entry.get('authors', [])
                    if not authors:
                        writer.writerow({'filename': filename, 'author_name': '', 'author_title': '', 'author_email': ''})
                    else:
                        for author in authors:
                            writer.writerow({
                                'filename': filename,
                                'author_name': author.get('name', ''),
                                'author_title': author.get('title', ''),
                                'author_email': author.get('email', '')
                            })
            logging.info(f"Appended {len(new_results)} records to {output_csv} (long layout).")
        except Exception as e:
            logging.error(f"Error appending to CSV file {output_csv}: {e}")
            raise
        return

    # ------------------------------------------------------------------
    # Wide layout handling
    # ------------------------------------------------------------------
    file_exists = os.path.exists(output_csv)
    headers = ['filename']
    for i in range(1, max_authors + 1):
        headers.extend([f'author_{i}_name', f'author_{i}_title', f'author_{i}_email'])
    try:
        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            for entry in new_results:
                row = {'filename': entry.get('filename', '')}
                for i in range(1, max_authors + 1):
                    if i <= len(entry.get('authors', [])):
                        author = entry['authors'][i-1]
                        row[f'author_{i}_name'] = author.get('name', '')
                        row[f'author_{i}_title'] = author.get('title', '')
                        row[f'author_{i}_email'] = author.get('email', '')
                    else:
                        row[f'author_{i}_name'] = ''
                        row[f'author_{i}_title'] = ''
                        row[f'author_{i}_email'] = ''
                writer.writerow(row)
        logging.info(f"Appended {len(new_results)} new results to {output_csv} (wide layout).")
    except Exception as e:
        logging.error(f"Error appending to CSV file {output_csv}: {e}")
        raise
    """
    Append new results to the CSV file instead of rewriting the entire file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(output_csv)
    
    # Create headers list
    headers = ['filename']
    for i in range(1, max_authors + 1):
        headers.extend([f'author_{i}_name', f'author_{i}_title', f'author_{i}_email'])
    
    try:
        # Open in append mode if file exists, otherwise write mode
        mode = 'a' if file_exists else 'w'
        with open(output_csv, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            # Only write header if it's a new file
            if not file_exists:
                writer.writeheader()
                
            # Write new data
            for entry in new_results:
                row = {'filename': entry.get('filename', '')}
                authors = entry.get('authors', [])
                for i in range(1, max_authors + 1):
                    if i <= len(authors):
                        author = authors[i-1]
                        row[f'author_{i}_name'] = author.get('name', '')
                        row[f'author_{i}_title'] = author.get('title', '')
                        row[f'author_{i}_email'] = author.get('email', '')
                    else:
                        row[f'author_{i}_name'] = ''
                        row[f'author_{i}_title'] = ''
                        row[f'author_{i}_email'] = ''
                writer.writerow(row)
        logging.info(f"Appended {len(new_results)} new results to {output_csv}")
    except Exception as e:
        logging.error(f"Error appending to CSV file {output_csv}: {e}")
        raise

def process_file_with_timeout(file_path, original_filename, server_url, results_list, results_lock):
    """
    Process a single file and add results to the results list.
    This function is designed to be called in a separate thread with a timeout.
    """
    try:
        result = process_single_file(file_path, original_filename, server_url)
        with results_lock:
            results_list.append(result)
        return True
    except Exception as e:
        logging.error(f"Error in timeout thread processing {original_filename}: {e}")
        return False


def process_directory(input_dir: str, output_csv: str, max_files: int = None, active_servers: List[str] = None) -> None:
    """
    Process text files in the given directory using multiple Ollama servers concurrently.
    Existing processed files (based on CSV records) are skipped.
    """
    # Use the active_servers list if provided
    servers_to_use = active_servers
    start_time = time.time()
    
    logging.info(f"Using {len(servers_to_use)} Ollama servers for processing")
    
    # Only load filenames to reduce memory usage
    processed_filenames, current_max_authors = load_processed_filenames(output_csv)
    new_files = get_unprocessed_files(input_dir, processed_filenames, max_files)
    total_files = len(new_files)

    if not new_files:
        logging.info("No new files to process.")
        return
    
    logging.info(f"Found {total_files} new files to process")

    # Create a thread-safe queue for file processing tasks
    task_queue = queue.Queue()
    for file_path, original_filename in new_files:
        task_queue.put((file_path, original_filename))

    # Track cooldown periods for each server (if errors occur)
    server_cooldown = {server: 0 for server in servers_to_use}
    cooldown_lock = threading.Lock()
    results = []
    results_lock = threading.Lock()
    
    # Counters for progress tracking
    processed_count = 0
    processed_count_lock = threading.Lock()
    
    # Write results in batches to avoid memory buildup
    batch_size = 20  # Write every 20 files
    last_write_count = 0

    def worker(server_url: str) -> None:
        nonlocal task_queue, server_cooldown, processed_count, last_write_count
        while True:
            try:
                file_path, original_filename = task_queue.get_nowait()
            except queue.Empty:
                break

            current_time = time.time()
            with cooldown_lock:
                if server_cooldown[server_url] > current_time:
                    # Server is cooling down; requeue this file and pause before retrying.
                    task_queue.put((file_path, original_filename))
                    task_queue.task_done()
                    time.sleep(1)
                    continue

            # Add retry counter to track problematic files
            retry_attempts = getattr(task_queue.queue[-1], 'retry_count', 0) if not task_queue.empty() else 0
            max_retries = 3  # Maximum number of retries per file

            try:
                logging.info(f"Starting to process file: {original_filename} on {server_url}")
                
                # Set a timeout for processing this file
                processing_timeout = 60  # 1 minute timeout
                processing_thread = threading.Thread(
                    target=process_file_with_timeout,
                    args=(file_path, original_filename, server_url, results, results_lock)
                )
                processing_thread.daemon = True
                processing_thread.start()
                
                # Wait for the processing to complete with timeout
                start_time = time.time()
                while processing_thread.is_alive() and time.time() - start_time < processing_timeout:
                    time.sleep(0.5)  # Check every half second
                
                # Check if processing timed out
                if processing_thread.is_alive():
                    # Processing took too long - log it and requeue if under max retries
                    logging.warning(f"Processing of {original_filename} timed out after {processing_timeout} seconds")
                    
                    if retry_attempts < max_retries:
                        # Requeue the file with incremented retry count
                        task_item = (file_path, original_filename)
                        setattr(task_item, 'retry_count', retry_attempts + 1)
                        task_queue.put(task_item)
                        logging.warning(f"Requeued {original_filename} for retry (attempt {retry_attempts + 1}/{max_retries})")
                    else:
                        # Too many retries, log it and move on
                        logging.error(f"Skipping {original_filename} after {max_retries} failed attempts")
                        
                        # Add a placeholder result for skipped files
                        with results_lock:
                            results.append({'filename': re.sub(r'_pages\d+.*\.txt$', '', original_filename).rstrip('_'), 'authors': []})
                    
                    # Force a cooldown for this server
                    with cooldown_lock:
                        server_cooldown[server_url] = time.time() + 10  # 10 second cooldown
                    
                    task_queue.task_done()
                    continue
                
                # If we got here, processing completed normally
                with processed_count_lock:
                    processed_count += 1
                    current_processed = processed_count
                    
                    # Calculate and display progress statistics
                    elapsed = time.time() - start_time
                    files_per_second = processed_count / elapsed if elapsed > 0 else 0
                    percent_complete = (processed_count / total_files) * 100 if total_files > 0 else 0
                    
                    # Write results in batches to avoid memory issues
                    if processed_count >= last_write_count + batch_size:
                        with results_lock:
                            if results:
                                # Get results to write
                                batch_to_write = results.copy()
                                
                                # Calculate max authors for this batch
                                batch_max_authors = max((len(r['authors']) for r in batch_to_write), default=0)
                                batch_max_authors = max(current_max_authors, batch_max_authors)
                                
                                # Write batch to CSV
                                append_results_to_csv(output_csv, batch_to_write, batch_max_authors)
                                
                                # Clear results after writing
                                results.clear()
                                
                                logging.info(f"Batch of {len(batch_to_write)} files written to CSV")
                                last_write_count = processed_count
                
                logging.info(f"Processed {current_processed}/{total_files} files ({percent_complete:.1f}%) - "
                        f"Speed: {files_per_second:.2f} files/second - File: {original_filename}")
            except Exception as e:
                logging.error(f"Error processing {original_filename} on {server_url}: {e}")
                # If an error occurs, set a cooldown for the server.
                with cooldown_lock:
                    server_cooldown[server_url] = time.time() + 100  # cooldown period in seconds
                task_queue.put((file_path, original_filename))
            finally:
                task_queue.task_done()


    # Start one worker thread per Ollama server.
    threads = []
    for server in servers_to_use:
        thread = threading.Thread(target=worker, args=(server,))
        thread.start()
        threads.append(thread)

    # Wait until all tasks are processed.
    task_queue.join()
    for thread in threads:
        thread.join()

    # Write any remaining results
    if results:
        new_max_authors = max((len(r['authors']) for r in results), default=0)
        max_authors = max(current_max_authors, new_max_authors)
        append_results_to_csv(output_csv, results, max_authors)
        
    total_time = time.time() - start_time
    logging.info(f"Processing complete! Processed {processed_count}/{total_files} files in {total_time:.1f} seconds")
    if processed_count > 0:
        logging.info(f"Average processing speed: {processed_count/total_time:.2f} files/second")


def process_single_file(file_path: str, original_filename: str, server_url: str) -> Dict:
    """
    Process a single text file:
      - Cleans the filename.
      - Reads file content and splits into manageable chunks if needed.
      - Extracts authors using an Ollama server.
      - Aggregates author information from chunks.
    """
    # Clean the filename by removing any '_pages...' pattern.
    clean_name = re.sub(r'_pages\d+.*\.txt$', '', original_filename).rstrip('_')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Failed to read file {original_filename}: {e}")
        raise
    
    # Check if chunking is needed based on token count
    token_count = get_token_count(content)
    logging.info(f"File {original_filename} has approximately {token_count} tokens")
    
    # Get max tokens from config
    max_tokens = get_max_tokens()
    
    # If content is small enough, process as a single chunk
    if token_count <= max_tokens:
        authors = extract_authors(content, server_url)
    else:
        # Split content into chunks
        chunks = chunk_text(content)
        logging.info(f"Split {original_filename} into {len(chunks)} chunks")
        
        # Process each chunk
        chunk_authors = []
        for i, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {i+1}/{len(chunks)} of {original_filename}")
            chunk_result = extract_authors(chunk, server_url)
            chunk_authors.append(chunk_result)
        
        # Aggregate authors from all chunks (if authors were returned)
        if chunk_authors and isinstance(chunk_authors[0], list):
            authors = aggregate_author_results(chunk_authors)  # type: ignore
            return {'filename': clean_name, 'authors': authors}
        else:
            # Score mode – each chunk should have returned a dict. We'll just use the
            # first chunk's dictionary (they should be identical because the prompt
            # asks for full-document analysis).
            scores = chunk_authors[0] if chunk_authors else {}
            scores = scores if isinstance(scores, dict) else {}
            return {'filename': clean_name, **scores}

def extract_authors(content: str, server_url: str) -> List[Dict]:
    """
    Use the Ollama client API to analyze file content and extract author information.
    The analysis adheres to guidelines targeting paragraphs, email patterns, and other indicators.
    """
    client = Client(host=server_url)
    try:
        # Get config values
        config_data = get_config()
        # Get prompt template from config and replace placeholder with content
        prompt_template = config_data['prompt']['template']
        prompt = prompt_template.replace("{{TEXT_CONTENT}}", content)
        
        response = client.chat(
            model=config_data['ollama']['model'],
            format='json',
            messages=[
                {
                    'role': 'system',
                    'content': prompt
                },
                {
                    'role': 'user',
                    'content': f"DOCUMENT:\n{content}\nAUTHORS:"
                }
            ],
            options={'num_ctx': get_max_tokens(), 'temperature': 0.0, 'top_p': 0.9}
        )
        
        return parse_model_response(response)
    except Exception as e:
        logging.error(f"Ollama client error on {server_url}: {e}")
        raise

def parse_model_response(response):
    """Parse the LLM response according to the strategy defined in the config
    (section 'parsing'). Supports two strategies:
    - 'json': expects well-formed JSON;
    - 'regex': extracts using a regular expression with named capture groups.
    Returns a list of author dictionaries with keys 'name', 'title', 'email'."""
    try:
        cfg = get_config()
        p_cfg = cfg.get('parsing', {})
        parse_type = p_cfg.get('type', 'json').lower()

        # Raw content string
        content = response.get('message', {}).get('content', '').strip()

        # Shortcut: if the caller provided explicit field mapping, we'll treat the
        # response as a per-document dictionary instead of an authors list.  This
        # enables alternate configs (e.g. AI-washing) to coexist with the original
        # author-extraction pipeline without code duplication.
        field_mapping: Dict[str, str] = p_cfg.get('fields', {})

        # Strip markdown code fences like ```json ... ```
        if content.startswith('```'):
            content = re.sub(r'^```[a-zA-Z0-9]*\n', '', content)
            content = content.rstrip('`').rstrip()

        authors: List[Dict] = []
        skip_domains = [d.lower() for d in p_cfg.get('skip_domains', ['mergent.com'])]

        if parse_type == 'json':
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                logging.error("JSON parse error in model response – returning empty list/dict")
                return {} if field_mapping else []

            # If a field_mapping is supplied, extract the specified keys and return
            # a dictionary (score mode). Each mapping value can be a dotted path.
            if field_mapping:
                extracted: Dict[str, Any] = {}
                for csv_key, path in field_mapping.items():
                    parts = path.split('.')
                    node = data
                    for part in parts:
                        if isinstance(node, dict) and part in node:
                            node = node[part]
                        else:
                            node = ''
                            break
                    extracted[csv_key] = node
                return extracted

            authors_key = p_cfg.get('authors_key', 'authors')
            raw_authors = data.get(authors_key, [])
            if not isinstance(raw_authors, list):
                return []

            name_key = p_cfg.get('name_key', 'name')
            title_key = p_cfg.get('title_key', 'title')
            email_key = p_cfg.get('email_key', 'email')

            for a in raw_authors:
                if not isinstance(a, dict):
                    continue
                name = str(a.get(name_key, '')).strip()
                title = str(a.get(title_key, '')).strip() if isinstance(a.get(title_key), str) else ''
                email = str(a.get(email_key, '')).strip()

                if email and any(dom in email.lower() for dom in skip_domains):
                    continue

                if name and len(name.split()) >= 2 and any(c.isupper() for c in name) and len(name) <= 100:
                    authors.append({"name": name, "title": title, "email": email})

        elif parse_type == 'regex':
            pattern = p_cfg.get('regex_pattern', '')
            if not pattern:
                logging.error("Regex parsing selected but 'regex_pattern' is empty in config.")
                return []
            try:
                regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
            except re.error as ex:
                logging.error(f"Invalid regex pattern: {ex}")
                return []

            name_group = p_cfg.get('name_group', 'name')
            title_group = p_cfg.get('title_group', 'title')
            email_group = p_cfg.get('email_group', 'email')

            for m in regex.finditer(content):
                gd = m.groupdict()
                name = gd.get(name_group, '').strip()
                title = gd.get(title_group, '').strip()
                email = gd.get(email_group, '').strip()

                if email and any(dom in email.lower() for dom in skip_domains):
                    continue
                if name and len(name.split()) >= 2 and any(c.isupper() for c in name) and len(name) <= 100:
                    authors.append({"name": name, "title": title, "email": email})
        else:
            logging.error(f"Unsupported parse type: {parse_type}")
            return []

        return authors
    except Exception as e:
        logging.error(f"Error parsing model response: {e}")
        return []

def load_processed_filenames(output_csv: str) -> Tuple[Set[str], int]:
    """
    Only read the filenames and max_authors from CSV to avoid loading all data into memory.
    Still creates a backup of the existing CSV file with timestamp.
    Returns: processed filenames and current maximum number of authors.
    """
    processed_filenames = set()
    current_max_authors = 0

    if os.path.exists(output_csv):
        try:
            # Create backup directory if it doesn't exist
            backup_dir = os.path.join(os.path.dirname(output_csv), 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create a timestamped backup of the current CSV
            timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
            base_filename = os.path.basename(output_csv)
            backup_filename = f"{os.path.splitext(base_filename)[0]}_{timestamp}.csv"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy the existing CSV to the backup location
            import shutil
            shutil.copy2(output_csv, backup_path)
            logging.info(f"Created backup of existing CSV at {backup_path}")
            
            # Read the CSV file to extract filenames and determine max_authors
            with open(output_csv, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # Get headers
                
                # Find the max author number from headers
                author_columns = [col for col in headers if col.startswith('author_')]
                if author_columns:
                    # Extract numeric parts from author column headers
                    author_nums = []
                    for col in author_columns:
                        match = re.search(r'author_(\d+)_', col)
                        if match:
                            author_nums.append(int(match.group(1)))
                    if author_nums:
                        current_max_authors = max(author_nums)
                
                # Only read the filename column (first column)
                for row in reader:
                    if row and len(row) > 0:
                        filename = row[0].strip()
                        if filename:
                            processed_filenames.add(filename)
                
                logging.info(f"Found {len(processed_filenames)} previously processed files")
                logging.info(f"Current max authors in CSV: {current_max_authors}")
        except Exception as e:
            logging.error(f"Error reading CSV file {output_csv}: {e}")
    return processed_filenames, current_max_authors


def get_unprocessed_files(input_dir: str, processed_filenames: Set[str], max_files: int = None) -> List[Tuple[str, str]]:
    """
    Retrieve a list of new text files from the input directory while skipping files that
    have already been processed (the filename is cleaned for matching).
    """
    new_files = []
    try:
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                clean_name = re.sub(r'_pages\d+.*\.txt$', '', filename).rstrip('_')
                if clean_name not in processed_filenames:
                    file_path = os.path.join(input_dir, filename)
                    if os.path.isfile(file_path):
                        new_files.append((file_path, filename))
                        if max_files and len(new_files) >= max_files:
                            break
    except Exception as e:
        logging.error(f"Error listing directory {input_dir}: {e}")
    return new_files

def save_results(output_csv: str, data: List[Dict], max_authors: int) -> None:
    """Save results to CSV respecting layout defined in config ('wide' or 'long')."""
    layout = get_csv_layout()
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # If data does not contain 'authors', treat as generic score rows
    if data and 'authors' not in data[0]:
        headers = ['filename'] + [k for k in data[0].keys() if k != 'filename']
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
            logging.info(f"Results saved to {output_csv}.")
        except Exception as e:
            logging.error(f"Error saving CSV file {output_csv}: {e}")
            raise
        return

    # ------------------------------------------------------------------
    # Existing author-extraction logic
    # ------------------------------------------------------------------
    if layout == 'long':
        headers = ['filename', 'author_name', 'author_title', 'author_email']
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for entry in data:
                    filename = entry.get('filename', '')
                    authors = entry.get('authors', [])
                    if not authors:
                        writer.writerow({'filename': filename, 'author_name': '', 'author_title': '', 'author_email': ''})
                    else:
                        for author in authors:
                            writer.writerow({
                                'filename': filename,
                                'author_name': author.get('name', ''),
                                'author_title': author.get('title', ''),
                                'author_email': author.get('email', '')
                            })
            logging.info(f"Results saved to {output_csv} (long layout).")
        except Exception as e:
            logging.error(f"Error saving CSV file {output_csv}: {e}")
            raise
        return  # Finished long layout path

    # ------------------------------------------------------------------
    # Wide layout (default): one row per file, dynamic author columns
    # ------------------------------------------------------------------
    headers = ['filename']
    for i in range(max_authors):
        headers.extend([f'author_{i+1}_name', f'author_{i+1}_title', f'author_{i+1}_email'])
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for entry in data:
                row = {'filename': entry.get('filename', '')}
                for i in range(max_authors):
                    if i < len(entry.get('authors', [])):
                        author = entry['authors'][i]
                        row[f'author_{i+1}_name'] = author.get('name', '')
                        row[f'author_{i+1}_title'] = author.get('title', '')
                        row[f'author_{i+1}_email'] = author.get('email', '')
                    else:
                        row[f'author_{i+1}_name'] = ''
                        row[f'author_{i+1}_title'] = ''
                        row[f'author_{i+1}_email'] = ''
                writer.writerow(row)
        logging.info(f"Results saved to {output_csv} (wide layout, max_authors={max_authors}).")
    except Exception as e:
        logging.error(f"Error saving CSV file {output_csv}: {e}")
        raise
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    headers = ['filename']
    for i in range(max_authors):
        headers.extend([f'author_{i+1}_name', f'author_{i+1}_title', f'author_{i+1}_email'])
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for entry in data:
                row = {'filename': entry.get('filename', '')}
                authors = entry.get('authors', [])
                for i in range(max_authors):
                    if i < len(authors):
                        author = authors[i]
                        row[f'author_{i+1}_name'] = author.get('name', '')
                        row[f'author_{i+1}_title'] = author.get('title', '')
                        row[f'author_{i+1}_email'] = author.get('email', '')
                    else:
                        row[f'author_{i+1}_name'] = ''
                        row[f'author_{i+1}_title'] = ''
                        row[f'author_{i+1}_email'] = ''
                writer.writerow(row)
        logging.info(f"Results saved to {output_csv}.")
    except Exception as e:
        logging.error(f"Error saving CSV file {output_csv}: {e}")
        raise

def get_active_ollama_servers() -> List[str]:
    """
    Dynamically detect which Ollama servers are running and return their addresses.
    Uses a port scan on the expected Ollama port range from config.
    
    Returns:
        List[str]: List of active server addresses in format "127.0.0.1:PORT"
    """
    active_servers = []
    try:
        # Get config data
        config_data = get_config()
        # Get port range from config
        port_range = config_data['ollama']['port_range']
        
        # If auto-detect is disabled in config, return fallback_api_url
        if not config_data['ollama']['auto_detect']:
            return [config_data['ollama']['fallback_api_url']]
            
        for port in range(port_range[0], port_range[1] + 1):
            try:
                # Try to create a socket connection to check if port is open
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)  # Short timeout for quick scanning
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                
                if result == 0:  # Port is open
                    active_servers.append(f'127.0.0.1:{port}')
            except:
                pass
                
        # If no servers found but auto-detect is enabled, return fallback
        if not active_servers:
            logging.warning("No active Ollama servers detected, using fallback URL")
            active_servers.append(config_data['ollama']['fallback_api_url'])
    except Exception as e:
        logging.error(f"Error detecting Ollama servers: {e}")
        # Return fallback in case of error
        fallback_url = get_config()['ollama']['fallback_api_url']
        active_servers.append(fallback_url)
    
    return active_servers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process directory files and generate a CSV report.')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json file (default: use config.json in the script directory)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Output CSV filename (overrides config.json setting)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (overrides config.json setting)')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Target token count per chunk when splitting documents (overrides config.json setting)')
    parser.add_argument('--chunk_overlap', type=int, default=None,
                        help='Overlap between chunks to maintain context (overrides config.json setting)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging (overrides config.json setting)')
    args = parser.parse_args()
    
    # Load config with specified path if provided
    if args.config:
        # Load a new config from the specified file and store it in the function's static variable
        get_config._config = load_json_config(args.config)
    
    # Get a reference to the config for updating
    config_data = get_config()
    
    # Override config values with command line arguments if provided
    if args.output_csv:
        config_data['output']['csv_filename'] = args.output_csv
    if args.max_files:
        config_data['execution']['max_files'] = args.max_files
    if args.chunk_size:
        config_data['processing']['chunk_size'] = args.chunk_size
    if args.chunk_overlap:
        config_data['processing']['chunk_overlap'] = args.chunk_overlap
    if args.debug:
        config_data['debug']['enabled'] = True
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled from command line")

    # Get the output directory from config
    output_dir = config_data['output']['directory']

    # Auto-detect active Ollama servers
    active_servers = get_active_ollama_servers()
    if not active_servers:
        logging.error("No Ollama servers detected. Please start Ollama servers.")
        exit(1)
    logging.info(f"Auto-detected {len(active_servers)} running Ollama servers")
    logging.info(f"Active Ollama servers: {', '.join(active_servers)}")

    # Construct the full output CSV path by joining the directory with the filename
    output_csv_path = os.path.join(output_dir, config_data['output']['csv_filename'])
    
    process_directory(
        input_dir=config_data['input']['directory'],
        output_csv=output_csv_path,
        max_files=config_data['execution']['max_files'],
        active_servers=active_servers
    )
