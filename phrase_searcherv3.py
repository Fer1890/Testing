# Part 1: Imports, Configuration, and Logging

import asyncio
import json
import csv
import datetime
import logging
import os
import re
from pathlib import Path
from typing import List, Dict
from asyncio import Semaphore

from playwright.async_api import async_playwright
from tqdm import tqdm

def load_config() -> Dict:
    """Loads configuration from config.json."""
    config_path = Path('config.json')
    if not config_path.exists():
        print("Config file not found. Creating default 'config.json'.")
        default_config = {
            "input_urls_file": "input_urls.csv",
            "input_phrases_file": "search_phrases.csv",
            "concurrent_limit": 2, 
            "timeout": 30000,
            "max_retries": 3,
            "retry_delay": 2,
            "save_batch_size": 2000,
            "debug_mode": False
        }
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config
    
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(log_dir: Path, debug_mode: bool) -> logging.Logger:
    """Sets up file and console logging."""
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger('AdvancedPhraseSearcher')
    logger.setLevel(logging.DEBUG) 
    
    if logger.hasHandlers():
        logger.handlers.clear()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f'search_log_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# Part 2: The AdvancedPhraseSearcher Class Structure

class AdvancedPhraseSearcher:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('AdvancedPhraseSearcher')
        
        # In-memory batch storage
        self.results = []
        self.errors = []
        
        # Persistent metrics across batches
        self.processed_urls_count = 0
        self.total_success_count = 0
        self.total_error_count = 0
        
        self.start_time = None
        self.semaphore = Semaphore(config.get('concurrent_limit', 1))
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("phrase_search_results") / timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Master file paths
        self.master_results_path = self.results_dir / f"results_master_{timestamp}.csv"
        self.master_errors_path = self.results_dir / f"errors_master_{timestamp}.csv"
        self.results_header_written = False
        self.errors_header_written = False
        
        self.playwright = None
        self.shared_context = None 

    async def initialize(self):
        """Initializes a PERSISTENT browser context that remembers cookies."""
        self.logger.info("Initializing Playwright and launching persistent browser...")
        self.playwright = await async_playwright().start()
        
        user_data_dir = os.path.join(os.getcwd(), 'robot_profile')
        
        # Set to True for pure background processing via your corporate network bypass
        self.shared_context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            channel="chrome",  
            headless=True, 
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True,
            java_script_enabled=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars'
            ]
        )
        self.logger.info("Persistent Context initialized successfully.")

    async def cleanup(self):
        """Closes the context and cleans up resources."""
        self.logger.info("Cleaning up Playwright resources...")
        if self.shared_context:
            await self.shared_context.close()
        if self.playwright:
            await self.playwright.stop()
        self.logger.info("Cleanup complete.")

    def read_urls_from_csv(self) -> List[Dict[str, str]]:
        urls_file = self.config['input_urls_file']
        urls = []
        try:
            with open(urls_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not {'url', 'page_name'}.issubset(reader.fieldnames or []):
                    raise ValueError("Input CSV must have 'url' and 'page_name' columns.")
                for row in reader:
                    if row['url'].strip() and row['page_name'].strip():
                        urls.append(row)
            self.logger.info(f"Successfully read {len(urls)} URLs from {urls_file}")
            return urls
        except FileNotFoundError:
            self.logger.error(f"Error: Input file '{urls_file}' not found.")
            return []
        except Exception as e:
            self.logger.error(f"Error reading URL file: {e}")
            return []

    def read_phrases_from_csv(self) -> List[str]:
        phrases_file = self.config['input_phrases_file']
        phrases = []
        try:
            with open(phrases_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    phrases.extend([phrase.strip() for phrase in row if phrase.strip()])
            return phrases
        except FileNotFoundError:
            self.logger.error(f"Error: Phrases file '{phrases_file}' not found.")
            return []
        except Exception as e:
            self.logger.error(f"Error reading phrases file: {e}")
            return []


# Part 3: Orchestration and Concurrent Processing

    async def process_urls(self, urls: List[Dict[str, str]], phrases: List[str]):
        self.start_time = datetime.datetime.now()
        total_urls = len(urls)
        
        with tqdm(total=total_urls, desc="Processing URLs", unit="url") as pbar:
            tasks = []
            for url_info in urls:
                task = asyncio.create_task(self._process_url_wrapper(url_info, phrases, pbar))
                tasks.append(task)
            await asyncio.gather(*tasks)

    async def _process_url_wrapper(self, url_info: Dict[str, str], phrases: List[str], pbar: tqdm):
        async with self.semaphore:
            try:
                result = await self.search_phrases_in_url(url_info, phrases)
                if result.get('error'):
                    self.errors.append(result)
                else:
                    self.results.append(result)
            except Exception as e:
                self.logger.error(f"Unhandled exception for {url_info['url']}: {e}")
                self.errors.append({
                    'page_name': url_info['page_name'],
                    'url': url_info['url'],
                    'error': str(e)
                })
            finally:
                self.processed_urls_count += 1
                if self.processed_urls_count % self.config['save_batch_size'] == 0:
                    self.flush_results_to_disk()
                pbar.update(1)

    def flush_results_to_disk(self):
        """Appends current memory batches to the master CSVs and frees memory."""
        if self.results:
            self._save_to_csv(self.master_results_path, self.results, is_error_file=False)
            self.total_success_count += len(self.results)
            self.results.clear()

        if self.errors:
            self._save_to_csv(self.master_errors_path, self.errors, is_error_file=True)
            self.total_error_count += len(self.errors)
            self.errors.clear()


# Part 4: Intelligent Content Extraction and Searching

    async def search_phrases_in_url(self, url_info: Dict[str, str], phrases: List[str]) -> Dict:
        url = url_info['url']
        page_name = url_info['page_name']
        page = None
        
        for attempt in range(self.config['max_retries']):
            try:
                page = await self.shared_context.new_page()
                await page.goto(url, wait_until='load', timeout=self.config['timeout'])
                
                # Dynamic wait for the body element to attach
                await page.wait_for_selector('body', state='attached', timeout=10000)
                
                # Read directly from the DOM, bypassing CSS visibility rules
                main_text = await self._get_filtered_page_text(page)
                
                phrase_results = {}
                for phrase in phrases:
                    phrase_results[phrase] = self._search_phrase_in_text(main_text, phrase)

                await page.close()
                return {
                    'page_name': page_name,
                    'url': url,
                    'phrase_results': phrase_results,
                    'error': None
                }

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{self.config['max_retries']} failed for {url}: {e}")
                if page:
                    await page.close()
                if attempt == self.config['max_retries'] - 1:
                    return {
                        'page_name': page_name,
                        'url': url,
                        'error': str(e)
                    }
                await asyncio.sleep(self.config['retry_delay'] * (2 ** attempt))

    async def _get_filtered_page_text(self, page) -> str:
        """Extracts raw textContent directly from the DOM, stripping scripts/styles."""
        try:
            text = await page.evaluate("""() => {
                // Clone to avoid mutating the live rendering engine
                const clone = document.body.cloneNode(true);
                const scriptsAndStyles = clone.querySelectorAll('script, style, noscript');
                scriptsAndStyles.forEach(el => el.remove());
                return clone.textContent;
            }""")
            return ' '.join(text.split()) if text else ""
        except Exception as e:
            self.logger.warning(f"Error extracting DOM text: {e}")
            return ""

    def _search_phrase_in_text(self, text: str, phrase: str) -> Dict:
        """Handles both explicit REGEX patterns and literal string matches."""
        occurrences = 0
        contexts = []
        
        # Check if the user wants to use a Regex pattern
        if phrase.startswith("REGEX:"):
            search_pattern = phrase.replace("REGEX:", "", 1).strip()
        else:
            # Standard exact phrase search (escaped for safety)
            search_pattern = re.escape(phrase)
        
        try:
            # Find all matches, case-insensitive
            for match in re.finditer(search_pattern, text, re.IGNORECASE):
                occurrences += 1
                if len(contexts) < 5: 
                    start_index = match.start()
                    end_index = match.end()
                    
                    context_start = max(0, start_index - 100)
                    context_end = min(len(text), end_index + 100)
                    context_snippet = text[context_start:context_end].strip()
                    contexts.append(f"...{' '.join(context_snippet.split())}...")
        except re.error as e:
            self.logger.error(f"Invalid Regex pattern '{search_pattern}': {e}")

        return {
            'found': occurrences > 0,
            'occurrences': occurrences,
            'contexts': contexts
        }
    

# Part 5: Saving Results and Main Execution

    def save_results(self):
        self.logger.info("Saving final results...")
        self.flush_results_to_disk()
        self._save_summary()
        self.logger.info(f"All results saved to: {self.results_dir}")

    def _save_to_csv(self, path: Path, data: List[Dict], is_error_file: bool):
        if not data: return
        try:
            mode = 'a' if path.exists() else 'w'
            with open(path, mode, newline='', encoding='utf-8') as f:
                if is_error_file:
                    fieldnames = ['page_name', 'url', 'error']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    if not self.errors_header_written:
                        writer.writeheader()
                        self.errors_header_written = True
                    writer.writerows(data)
                else:
                    first_result = data[0]
                    phrases = list(first_result['phrase_results'].keys())
                    base_fields = ['page_name', 'url']
                    phrase_fields = sum([[f'"{p}" Found', f'"{p}" Occurrences', f'"{p}" Contexts'] for p in phrases], [])
                    fieldnames = base_fields + phrase_fields
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not self.results_header_written:
                        writer.writeheader()
                        self.results_header_written = True
                        
                    for row in data:
                        flat_row = {'page_name': row['page_name'], 'url': row['url']}
                        for phrase, presults in row['phrase_results'].items():
                            flat_row[f'"{phrase}" Found'] = 'YES' if presults['found'] else 'NO'
                            flat_row[f'"{phrase}" Occurrences'] = presults['occurrences']
                            flat_row[f'"{phrase}" Contexts'] = ' | '.join(presults['contexts'])
                        writer.writerow(flat_row)
        except Exception as e:
            self.logger.error(f"Failed to save CSV to {path}: {e}")

    def _save_summary(self):
        summary_path = self.results_dir / "summary.txt"
        end_time = datetime.datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() if self.start_time else 0

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Phrase Search Summary\n")
            f.write("="*25 + "\n")
            f.write(f"Report Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Overall Statistics ##\n")
            f.write(f"Total URLs Processed: {self.processed_urls_count}\n")
            f.write(f"Total Successful URLs: {self.total_success_count}\n")
            f.write(f"Total Failed URLs: {self.total_error_count}\n\n")
            f.write("## Performance Metrics ##\n")
            f.write(f"Total Execution Time: {total_duration:.2f} seconds\n")
            if self.processed_urls_count > 0:
                avg_time = total_duration / self.processed_urls_count
                f.write(f"Average Time per URL: {avg_time:.2f} seconds\n")


# --- Main Execution ---

async def main():
    config = load_config()
    logger = setup_logging(Path("logs"), config.get('debug_mode', False))
    searcher = AdvancedPhraseSearcher(config)
    
    try:
        urls = searcher.read_urls_from_csv()
        phrases = searcher.read_phrases_from_csv()
        
        if not urls:
            logger.error("No URLs loaded. Exiting.")
            return
        if not phrases:
            logger.error("No phrases loaded. Exiting.")
            return
            
        await searcher.initialize()
        await searcher.process_urls(urls, phrases)
        searcher.save_results()
        
    except Exception as e:
        logger.error(f"A fatal error occurred in main execution: {e}", exc_info=True)
    finally:
        await searcher.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    finally:
        print("\nScript finished.")