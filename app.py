#!/usr/bin/env python3
import os
import sys
import re
import venv
import logging
import subprocess
import traceback
import time
from itertools import zip_longest
from io import StringIO
from pathlib import Path
from datetime import datetime
from git import Repo, InvalidGitRepositoryError
import html
import json
import requests
from urllib.parse import urlparse

try:
    from openai import OpenAI
    import gradio as gr
except ImportError as e:
    print(f"Required package not found: {e}. Will be installed in venv setup.")
    
# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"llm_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Ensure terminal encoding is UTF-8
try:
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
except Exception as e:
     print(f"Could not set stdout to utf8: {e}")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("LLMInterface")

def setup_venv():
    """Create and activate virtual environment."""
    logger.info("Setting up virtual environment...")

    venv_dir = Path(".venv")
    if not venv_dir.exists():
        logger.info("Creating new virtual environment...")
        venv.create(venv_dir, with_pip=True)

    if sys.platform == "win32":
        python_path = venv_dir / "Scripts" / "python.exe"
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        python_path = venv_dir / "bin" / "python"
        pip_path = venv_dir / "bin" / "pip"

    # Install required packages
    try:
        logger.info("Installing required packages...")
        subprocess.check_call([str(pip_path), "install", "gradio", "openai", "GitPython"])
        logger.info("Package installation successful")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        raise

    return str(python_path)

def restart_in_venv():
    """Ensure running in virtual environment."""
    if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
        logger.info("Not running in venv, setting up and restarting...")
        python_path = setup_venv()
        os.execv(python_path, [python_path] + sys.argv)

class ExecutionManager:
    """Manages code execution and diffs."""

    def __init__(self, security_manager):
        self.security_manager = security_manager
        self.repo = self.get_git_repo()
        self.last_code = None
        self.last_output = None

    def get_git_repo(self):
        """Get the Git repository or None if not a Git repo."""
        try:
            repo = Repo(search_parent_directories=True)
            return repo
        except InvalidGitRepositoryError:
            logger.info("Not a Git repository.")
            return None

    def update_last_code_and_output(self, code, output):
        """Updates the last executed code and output."""
        self.last_code = code
        self.last_output = output

    def get_last_code_html(self):
        """Returns the last executed code as HTML."""
        if self.last_code:
            escaped_code = html.escape(self.last_code)
            return f"<pre><code>{escaped_code}</code></pre>"
        else:
            return "<p>No code executed yet.</p>"

    def get_last_output_html(self):
        """Returns the last output as HTML."""
        if self.last_output:
            escaped_output = html.escape(self.last_output)
            return f"<pre>{escaped_output}</pre>"
        else:
            return "<p>No output yet.</p>"

class SecurityManager:
    def __init__(self):
        # Patterns that are always blocked
        self.blocked_patterns = [
            r"rm\s+-rf\s+/",
            r"system\s*\(",
            r"(?<!@)eval\s*\(",
            r"(?<!controlled_)exec\s*\(",
            r"subprocess\.",
            r"pty\.",
            r"__import__\s*\(",
            r"globals\s*\(\s*\)",
            r"locals\s*\(\s*\)",
            r"breakpoint\s*\(\s*\)",
            r"input\s*\("
        ]
        
        # Patterns that require extra validation
        self.restricted_patterns = {
            r"requests\.": self._validate_requests,
            r"urlopen\s*\(": self._validate_url,
            r"open\s*\(": self._validate_file_access,
            r"Path\s*\(": self._validate_path
        }
        
        # Allowed domains for web requests
        self.allowed_domains = {
            'api.github.com',
            'api.search.brave.com',
            'raw.githubusercontent.com',
            'api.openai.com',
            'api.serpapi.com',
            'arxiv.org',
            'en.wikipedia.org',
            'www.reddit.com',
            'scholar.google.com'
        }
        
        # Allowed file extensions
        self.allowed_extensions = {
            '.txt', '.csv', '.json', '.yaml', '.yml', 
            '.md', '.py', '.js', '.html', '.css'
        }
        
        # Initialize safe execution environment
        self.safe_globals = {
            'print': print,
            'len': len,
            'range': range,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'sum': sum,
            'min': min,
            'max': max,
            'enumerate': enumerate,
            'zip': zip,
            'round': round,
            'abs': abs,
            'all': all,
            'any': any,
            'sorted': sorted,
            'filter': filter,
            'map': map,
            'controlled_exec': self.controlled_exec,
            'os': self._create_safe_os(),
            'Path': Path,
            'requests': self._create_safe_requests(),
            'math': __import__('math'),
            'statistics': __import__('statistics'),
            'random': __import__('random'),
            'json': __import__('json'),
            'csv': __import__('csv'),
            're': __import__('re'),
            'datetime': __import__('datetime'),
            'itertools': __import__('itertools'),
            'collections': __import__('collections'),
            'os.stat': os.stat
        }

    def _create_safe_requests(self):
        """Create a restricted requests module with only GET methods to allowed domains"""
        class SafeRequests:
            def __init__(self, allowed_domains):
                self.allowed_domains = allowed_domains
                self.session = requests.Session()
                
            def get(self, url, **kwargs):
                domain = urlparse(url).netloc
                if domain not in self.allowed_domains:
                    raise SecurityError(f"Access to domain {domain} is not allowed")
                return self.session.get(url, **kwargs)
                
        return SafeRequests(self.allowed_domains)

    def _create_safe_os(self):
        """Create a restricted os module with only safe operations"""
        safe_os = type('SafeOS', (), {})()

        safe_functions = [
            'getcwd', 'listdir', 'path.exists', 'path.isfile',
            'path.isdir', 'path.getsize', 'path.basename',
            'path.dirname', 'path.join', 'path.splitext',
            'scandir'
        ]

        for func in safe_functions:
            if '.' in func:
                module, method = func.split('.')
                if not hasattr(safe_os, module):
                    setattr(safe_os, module, type('SafeOSPath', (), {})())
                setattr(getattr(safe_os, module), method, getattr(os.path, method))
            else:
                setattr(safe_os, func, getattr(os, func))

        return safe_os

    def _validate_requests(self, code_block):
        """Validate web requests"""
        urls = re.findall(r'(?:http[s]?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&\'\(\)\*\+,;=.]+', code_block)
        
        for url in urls:
            domain = urlparse(url).netloc
            if domain not in self.allowed_domains:
                raise SecurityError(f"Access to domain {domain} is not allowed")
        return True

    def _validate_url(self, code_block):
        """Validate URL operations"""
        return self._validate_requests(code_block)

    def _validate_file_access(self, code_block):
        """Validate file operations"""
        potential_files = re.findall(r'open\s*\(\s*[\'"]([^\'"]+)[\'"]', code_block)
        
        for file_path in potential_files:
            path = Path(file_path)
            if not any(str(path).endswith(ext) for ext in self.allowed_extensions):
                raise SecurityError(f"Access to file type {path.suffix} is not allowed")
            if not path.is_relative_to(Path.cwd()):
                raise SecurityError("Access to files outside working directory is not allowed")
        return True

    def _validate_path(self, code_block):
        """Validate Path operations"""
        return self._validate_file_access(code_block)

    def controlled_exec(self, code_str, globals_dict=None, locals_dict=None):
        """Safe version of exec with restricted scope"""
        if globals_dict is None:
            globals_dict = self.safe_globals.copy()
        return exec(code_str, globals_dict, locals_dict)

    def validate_code(self, code_block):
        """Validate code block against security rules"""
        try:
            for pattern in self.blocked_patterns:
                if re.search(pattern, code_block):
                    return False, f"Blocked pattern detected: {pattern}"

            for pattern, validator in self.restricted_patterns.items():
                if re.search(pattern, code_block):
                    try:
                        validator(code_block)
                    except SecurityError as e:
                        return False, str(e)

            return True, "Code passes security checks"

        except Exception as e:
            return False, f"Security validation error: {str(e)}"

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

class LLMManager:
    def __init__(self, execution_manager, 
                 model_a_url=None, 
                 model_b_url=None, 
                 model_a_alias="model_a", 
                 model_b_alias="model_b", 
                 model_a_id=None, 
                 model_b_id=None, 
                 max_tokens=2000, 
                 temperature=0.7, 
                 top_p=0.95, 
                 api_key=None):
        logger.info("Initializing LLMManager...")
        self.execution_manager = execution_manager
        self.security_manager = execution_manager.security_manager

        # Initialize LLM URLs and IDs from environment variables or defaults
        self.model_urls = {
            model_a_alias: model_a_url or os.getenv("MODEL_A_URL", "https://api.openai.com/v1"),
            model_b_alias: model_b_url or os.getenv("MODEL_B_URL", "https://api.openai.com/v1")
        }
        self.model_ids = {
            model_a_alias: model_a_id or os.getenv("MODEL_A_ID", "gpt-4"),
            model_b_alias: model_b_id or os.getenv("MODEL_B_ID", "gpt-4")
        }
        
        # Set API key from parameter or environment
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.warning("No API key provided. Please set API key in the configuration.")
        
        self.model_a_alias = model_a_alias
        self.model_b_alias = model_b_alias

        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "2000"))
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.7"))
        self.top_p = top_p or float(os.getenv("TOP_P", "0.95"))

        self._update_api_clients()

        # Track execution context and test passes
        self.last_execution_locals = {}
        self.passed_tests_count = 0
        self.max_passed_tests = 4

        # Updated system message for simpler syntax
        self.system_message = {
            "role": "system",
            "content": """You are an AI assistant with Python code execution capabilities.

Code execution and testing format:

1. Regular Python code:
```python
your_code_here
```

2. Test assertions:
```python-test
assert condition, "Test message"
```

The code blocks are executed in order. Variables from previous code blocks are available in subsequent blocks.
Generation stops after sufficient test passes or when requested.

Example:
Let me create a function and test it.

```python
def add(a, b):
    return a + b
result = add(5, 7)
print(f'Result: {result}')
```

```python-test
assert result == 12, "Addition should work"
assert add(-1, 1) == 0, "Should handle negatives"
```
"""
        }
        self.conversation = [self.system_message]

    def _update_api_clients(self):
        """Updates the api clients with the current config"""
        try:
            self.llama_api_clients = {}
            # Update the environment variable for OpenAI
            os.environ["OPENAI_API_KEY"] = self.api_key
            
            for alias, url in self.model_urls.items():
                self.llama_api_clients[alias] = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=url
                )
                logger.info(f"Initialized API client for {alias} with URL {url}")
        except Exception as e:
            logger.error(f"Failed to initialize LLMManager: {e}")
            raise

    def update_config(self, model_a_url=None, model_b_url=None, model_a_alias=None, model_b_alias=None,
                     model_a_id=None, model_b_id=None, max_tokens=None, temperature=None, top_p=None, api_key=None):
        """Updates the config of the LLM Manager"""
        if api_key:
            self.api_key = api_key
            logger.info("API key updated")
            
        if model_a_url and model_a_alias:
            self.model_urls[model_a_alias] = model_a_url
        if model_b_url and model_b_alias:
            self.model_urls[model_b_alias] = model_b_url
        if model_a_alias:
            self.model_a_alias = model_a_alias
        if model_b_alias:
            self.model_b_alias = model_b_alias
        if model_a_id and model_a_alias:
            self.model_ids[model_a_alias] = model_a_id
        if model_b_id and model_b_alias:
            self.model_ids[model_b_alias] = model_b_id
        if max_tokens:
            self.max_tokens = int(max_tokens)
        if temperature:
            self.temperature = float(temperature)
        if top_p:
            self.top_p = float(top_p)

        self._update_api_clients()
        logger.info(f"LLM Config updated to: MODEL_URLS: {self.model_urls}, MODEL_IDS: {self.model_ids} MODEL_A_ALIAS: {self.model_a_alias}, MODEL_B_ALIAS: {self.model_b_alias}, MAX_TOKENS: {self.max_tokens}, TEMPERATURE: {self.temperature}, TOP_P: {self.top_p}")

    def run_code(self, code, execution_locals):
        """Execute code with enhanced safety checks and diff tracking."""
        logger.info("Preparing to execute code block")
        logger.debug(f"Code to execute:\n{code}")
        
        # Validate code against security rules
        is_safe, message = self.security_manager.validate_code(code)
        if not is_safe:
            error_msg = f"Security check failed: {message}"
            logger.warning(error_msg)
            return error_msg

        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Create a new locals dict if none provided
            if execution_locals is None:
                execution_locals = {}

            # Execute code with safe globals
            self.security_manager.controlled_exec(code, self.security_manager.safe_globals, execution_locals)
            
            # Update the last execution locals
            self.last_execution_locals = execution_locals

            output = captured_output.getvalue()
            logger.info("Code execution successful")

            # Update last executed code and output
            self.execution_manager.update_last_code_and_output(code, output)

            return output
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Code execution failed: {str(e)}\n{error_trace}")
            return f"Error executing code:\n{error_trace}"
        finally:
            sys.stdout = old_stdout

    def run_tests(self, test_code):
        """Execute test assertions with access to previous code context."""
        logger.info("run_tests method called")
        logger.debug(f"Test code:\n{test_code}")

        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Include previous execution context
            test_globals = {
                'print': print,
                'assert': assert_,
                **self.last_execution_locals
            }

            exec(test_code, test_globals, {})
            output = captured_output.getvalue()
            logger.info(f"Tests passed")

            # Update last executed code and output
            self.execution_manager.update_last_code_and_output(test_code, output)

            return f"Tests passed{': ' + output if output else ''}"
        except AssertionError as e:
            logger.info(f"Test failed: {str(e)}")
            return f"Test failed: {str(e)}"
        except Exception as e:
            logger.error(f"Error running tests: {str(e)}")
            return f"Error running tests: {str(e)}"
        finally:
            sys.stdout = old_stdout

    def should_stop_generation(self):
        """Check if enough tests have passed to stop generation."""
        return self.passed_tests_count >= self.max_passed_tests

    def query_llama(self, model_alias, messages, stream=False):
        """Query the LLM model with streaming support."""
        logger.info(f"Querying model: {model_alias}")
        try:
            api_client = self.llama_api_clients.get(model_alias)
            if not api_client:
                raise ValueError(f"No api client found for alias {model_alias}")

            model_id = self.model_ids.get(model_alias)
            if not model_id:
                raise ValueError(f"No model id found for alias {model_alias}")

            chat_completion = api_client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream,
                top_p=self.top_p,
                presence_penalty=0,
                frequency_penalty=0
            )

            if stream:
                for chunk in chat_completion:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta and hasattr(chunk.choices[0].delta, 'content'):
                            content = chunk.choices[0].delta.content
                        elif hasattr(chunk.choices[0], 'text'):
                            content = chunk.choices[0].text
                        else:
                            content = ""
                    else:
                        content = ""

                    if content is not None:
                        yield content
            else:
                return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            error_msg = f"Error querying model {model_alias}: {str(e)}"
            logger.error(error_msg)
            if stream:
                yield f"Error: {error_msg}"
            else:
                return f"Error: {error_msg}"

    def process_message(self, message):
        """Process a user message with simplified code block execution."""
        logger.info("Processing new user message")
        self.passed_tests_count = 0  # Reset test counter

        if not message.strip():
            logger.warning("Empty message received")
            yield "Please enter a message", "Empty message received", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
            return

        try:
            self.conversation.append({"role": "user", "content": message})

            # --- Process Model A ---
            logger.info("Getting Model A response")
            response_a = ""

            try:
                for chunk in self.query_llama(self.model_a_alias, self.conversation, stream=True):
                    response_a += chunk
                    temp_conversation = self.get_conversation_history() + f"\n{self.model_a_alias}: {response_a}\n\n"
                    yield temp_conversation, "Processing Model A response...", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
            except Exception as e:
                error_msg = f"Error getting Model A response: {str(e)}"
                logger.error(error_msg)
                yield self.get_conversation_history(), error_msg, self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                return

            self.conversation.append({"role": "assistant", "name": self.model_a_alias, "content": response_a})

            # Process code blocks with simpler regex
            code_blocks = re.findall(r'```([^\n]+)\n(.*?)```', response_a, re.DOTALL)

            if code_blocks:
                for i, (lang, code) in enumerate(code_blocks):
                    lang = lang.strip().lower()
                    
                    if lang == 'python':
                        # Execute Python code
                        logger.info(f"Executing Python code block {i+1}")
                        output = self.run_code(code.strip(), self.last_execution_locals)
                        code_response = f"Output from block {i+1}:\n{output}"
                        self.conversation.append({"role": "assistant", "name": self.model_a_alias, "content": code_response})
                        yield self.get_conversation_history(), f"Executed code block {i+1}", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                        time.sleep(0.05)
                        
                    elif lang == 'python-test' or lang == 'pytest':
                        # Execute test code
                        logger.info(f"Executing test block {i+1}")
                        test_result = self.run_tests(code.strip())
                        self.conversation.append({"role": "assistant", "name": self.model_a_alias, "content": f"Test results:\n{test_result}"})
                        yield self.get_conversation_history(), f"Executed test block {i+1}", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                        time.sleep(0.05)

                        if "Test failed" not in test_result:
                            self.passed_tests_count += 1
                            if self.should_stop_generation():
                                logger.info("Required test passes achieved")
                                yield self.get_conversation_history(), "Complete", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                                return

            # Only proceed to Model B if needed
            if not self.should_stop_generation():
                logger.info("Getting Model B response")
                response_b = ""
                
                try:
                    conversation_context = self.get_conversation_history()
                    for chunk in self.query_llama(self.model_b_alias, self.conversation + [{"role": "user", "content": f"Current conversation: {conversation_context}"}], stream=True):
                        response_b += chunk
                        temp_conversation = self.get_conversation_history() + f"\n{self.model_b_alias}: {response_b}\n\n"
                        yield temp_conversation, "Processing Model B response...", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                except Exception as e:
                    error_msg = f"Error getting Model B response: {str(e)}"
                    logger.error(error_msg)
                    yield self.get_conversation_history(), error_msg, self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                    return

                self.conversation.append({"role": "assistant", "name": self.model_b_alias, "content": response_b})

                # Process Model B code blocks
                code_blocks = re.findall(r'```([^\n]+)\n(.*?)```', response_b, re.DOTALL)

                if code_blocks:
                    for i, (lang, code) in enumerate(code_blocks):
                        lang = lang.strip().lower()
                        
                        if lang == 'python':
                            logger.info(f"Executing Python code block {i+1} from Model B")
                            output = self.run_code(code.strip(), self.last_execution_locals)
                            code_response = f"Output from block {i+1}:\n{output}"
                            self.conversation.append({"role": "assistant", "name": self.model_b_alias, "content": code_response})
                            yield self.get_conversation_history(), f"Executed code block {i+1}", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                            time.sleep(0.05)
                            
                        elif lang == 'python-test' or lang == 'pytest':
                            logger.info(f"Executing test block {i+1} from Model B")
                            test_result = self.run_tests(code.strip())
                            self.conversation.append({"role": "assistant", "name": self.model_b_alias, "content": f"Test results:\n{test_result}"})
                            yield self.get_conversation_history(), f"Executed test block {i+1}", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                            time.sleep(0.05)

                            if "Test failed" not in test_result:
                                self.passed_tests_count += 1
                                if self.should_stop_generation():
                                    logger.info("Required test passes achieved")
                                    yield self.get_conversation_history(), "Complete", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()
                                    return

            yield self.get_conversation_history(), "Completed", self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            yield self.get_conversation_history(), error_msg, self.execution_manager.get_last_code_html(), self.execution_manager.get_last_output_html()

    def get_conversation_history(self):
        """Get formatted conversation history."""
        try:
            history = ""
            for msg in self.conversation[1:]:  # Skip system message
                role = msg.get("name", msg["role"])
                content = msg["content"]
                history += f"\n{role}: {content}\n"
            return history
        except Exception as e:
            error_msg = f"Error getting conversation history: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def clear_conversation(self):
        """Clear conversation history while preserving system message."""
        try:
            system_message = self.conversation[0]
            self.conversation = [system_message]
            self.passed_tests_count = 0
            self.last_execution_locals = {}
            logger.info("Conversation cleared")
            return "Conversation cleared."
        except Exception as e:
            error_msg = f"Error clearing conversation: {str(e)}"
            logger.error(error_msg)
            return error_msg

def create_ui():
    """Create and configure the Gradio interface."""
    logger.info("Creating Gradio interface")

    try:
        security_manager = SecurityManager()
        execution_manager = ExecutionManager(security_manager)
        
        # Initialize LLMManager with environment variables
        model_a_url = os.getenv("MODEL_A_URL", "http://127.0.0.1:1234/v1/")
        model_b_url = os.getenv("MODEL_B_URL", "http://127.0.0.1:1234/v1/")
        model_a_id = os.getenv("MODEL_A_ID", "phi-4")
        model_b_id = os.getenv("MODEL_B_ID", "phi-4")
        model_a_alias = "model_a"
        model_b_alias = "model_b"
        max_tokens = os.getenv("MAX_TOKENS", "2000")
        temperature = os.getenv("TEMPERATURE", "0.7")
        top_p = os.getenv("TOP_P", "0.95")
        api_key = os.getenv("API_KEY", "api_key")

        manager = LLMManager(execution_manager,
                           api_key=api_key,
                           model_a_url=model_a_url,
                           model_b_url=model_b_url,
                           model_a_alias=model_a_alias,
                           model_b_alias=model_b_alias,
                           model_a_id=model_a_id,
                           model_b_id=model_b_id,
                           max_tokens=int(max_tokens),
                           temperature=float(temperature),
                           top_p=float(top_p))

        def handle_update_env(model_a_url, model_b_url, model_a_id, model_b_id, 
                            model_a_alias, model_b_alias, max_tokens, temperature, top_p, api_key):
            """Handle the update of env variables"""
            try:
                manager.update_config(
                    model_a_url=model_a_url,
                    model_b_url=model_b_url,
                    model_a_id=model_a_id,
                    model_b_id=model_b_id,
                    model_a_alias=model_a_alias,
                    model_b_alias=model_b_alias,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    api_key=api_key
                )
                return (f"Configuration updated successfully. API key {'was' if api_key else 'was not'} updated. "
                       f"URLS: {manager.model_urls}, IDs: {manager.model_ids}, "
                       f"Aliases: {model_a_alias}/{model_b_alias}, "
                       f"Params: tokens={max_tokens}, temp={temperature}, top_p={top_p}"
                       ), model_a_url, model_b_url, model_a_id, model_b_id, model_a_alias, model_b_alias, max_tokens, temperature, top_p, api_key

            except Exception as e:
                error_msg = f"Error updating configuration: {str(e)}"
                logger.error(error_msg)
                return error_msg, model_a_url, model_b_url, model_a_id, model_b_id, model_a_alias, model_b_alias, max_tokens, temperature, top_p, api_key

        def handle_submit(message):
            """Handle message submission with streaming."""
            if not message:
                return "", "Please enter a message", "", ""

            try:
                logger.info(f"Handling new message: {message[:50]}...")
                result = manager.process_message(message)
                
                for response in result:
                    yield response

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                yield "", f"Error: {e}", execution_manager.get_last_code_html(), execution_manager.get_last_output_html()

        def handle_stop():
            """Handle stop button click."""
            return "Stopping generation...", "Stopping..."

        def handle_clear():
            """Handle conversation clearing."""
            try:
                result = manager.clear_conversation()
                return "", result, "<p>No code executed yet.</p>", "<p>No output yet.</p>"
            except Exception as e:
                error_msg = f"Error clearing conversation: {str(e)}"
                logger.error(error_msg)
                return "", error_msg, execution_manager.get_last_code_html(), execution_manager.get_last_output_html()

        def handle_show_last_code():
            """Handle show last code button click."""
            logger.info("handle_show_last_code called")
            return execution_manager.get_last_code_html()

        def handle_show_last_output():
            """Handle show last output button click."""
            logger.info("handle_show_last_output called")
            return execution_manager.get_last_output_html()

        with gr.Blocks(title="🚂🤖🪄 Conductor") as interface:
            gr.Markdown("# 🚂🤖🪄 Conductor")
            gr.Markdown("Enter your message to interact with the AI models. Code blocks are automatically executed, and tests must pass to continue.")

            with gr.Accordion("Environment Variables", open=False):
                with gr.Row():
                    model_a_url_input = gr.Textbox(
                        label="Model A URL",
                        value=model_a_url,
                        placeholder="http://127.0.0.1:1234/v1/"
                    )
                    model_b_url_input = gr.Textbox(
                        label="Model B URL",
                        value=model_b_url,
                        placeholder="http://127.0.0.1:1234/v1/"
                    )
                with gr.Row():
                    model_a_id_input = gr.Textbox(
                        label="Model A ID",
                        value=model_a_id,
                        placeholder="phi-4"
                    )
                    model_b_id_input = gr.Textbox(
                        label="Model B ID",
                        value=model_b_id,
                        placeholder="phi-4"
                    )
                with gr.Row():
                    model_a_alias_input = gr.Textbox(
                        label="Model A Alias",
                        value=model_a_alias,
                        placeholder="model_a"
                    )
                    model_b_alias_input = gr.Textbox(
                        label="Model B Alias",
                        value=model_b_alias,
                        placeholder="model_b"
                    )
                with gr.Row():
                    max_tokens_input = gr.Number(
                        label="Max Tokens",
                        value=int(max_tokens),
                    )
                    temperature_input = gr.Number(
                        label="Temperature",
                        value=float(temperature),
                    )
                    top_p_input = gr.Number(
                        label="Top P",
                        value=float(top_p),
                    )
                    api_key_input = gr.Textbox(
                        label="API Key",
                        value=api_key,
                        type="password"
                    )
                update_env_btn = gr.Button("Update Configuration")

            with gr.Row():
                with gr.Column(scale=2):
                    input_message = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Input Message",
                        lines=3
                    )

                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        stop_btn = gr.Button("Stop Generation", variant="secondary")
                        clear_btn = gr.Button("Clear Conversation")

                with gr.Column(scale=3):
                    conversation_display = gr.Textbox(
                        label="Conversation & Results",
                        lines=20,
                        interactive=False
                    )

            last_code_display = gr.HTML(
                label="Last Executed Code"
            )

            last_output_display = gr.HTML(
                label="Last Output"
            )

            with gr.Row():
                show_last_code_btn = gr.Button("Show Last Code")
                show_last_output_btn = gr.Button("Show Last Output")

            status_display = gr.Textbox(
                label="Status/Tests",
                lines=2,
                interactive=False,
                visible=True
            )

            # Wire up the interface events
            update_env_btn.click(
                fn=handle_update_env,
                inputs=[model_a_url_input, model_b_url_input, model_a_id_input, model_b_id_input,
                       model_a_alias_input, model_b_alias_input, max_tokens_input, temperature_input, 
                       top_p_input, api_key_input],
                outputs=[status_display, model_a_url_input, model_b_url_input, model_a_id_input, 
                        model_b_id_input, model_a_alias_input, model_b_alias_input, max_tokens_input, 
                        temperature_input, top_p_input]
            )

            submit_btn.click(
                fn=handle_submit,
                inputs=input_message,
                outputs=[conversation_display, status_display, last_code_display, last_output_display],
                show_progress=True
            )

            stop_btn.click(
                fn=handle_stop,
                inputs=None,
                outputs=[conversation_display, status_display]
            )

            clear_btn.click(
                fn=handle_clear,
                inputs=None,
                outputs=[conversation_display, status_display, last_code_display, last_output_display]
            )

            show_last_code_btn.click(
                fn=handle_show_last_code,
                inputs=None,
                outputs=last_code_display
            )

            show_last_output_btn.click(
                fn=handle_show_last_output,
                inputs=None,
                outputs=last_output_display
            )

            interface.load(
                fn=manager.get_conversation_history,
                inputs=None,
                outputs=conversation_display
            )

        interface.queue()
        return interface

    except Exception as e:
        error_msg = f"Error creating UI: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise

def main():
    """Main entry point."""
    logger.info("🚂🤖🪄 Initializing Conductor")

    try:
        # Ensure we're running in a virtual environment
        restart_in_venv()

        # Create and launch the interface
        interface = create_ui()
        logger.info("Launching Gradio interface")
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=31337,
            debug=True
        )

    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
