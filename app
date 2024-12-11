#!/usr/bin/env python3
import os
import sys
import re
import venv
import logging
import subprocess
import traceback
import time
from io import StringIO
from pathlib import Path
from datetime import datetime
from git import Repo, InvalidGitRepositoryError
import html

try:
    from openai import OpenAI
    import gradio as gr
except ImportError as e:
    logger.error(f"Required package not found: {e}. Will be installed in venv setup.")

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"llm_interface_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

class Versions:
    """Manages version control and code diffs."""
    def __init__(self):
        self.repo = self.get_git_repo()
        self.code_versions = {}  # Store escaped code versions

    def get_git_repo(self):
        """Get the Git repository or None if not a Git repo."""
        try:
            repo = Repo(search_parent_directories=True)
            return repo
        except InvalidGitRepositoryError:
            logger.info("Not a Git repository.")
            return None

    def capture_file_state(self):
        """Capture the current state of .py files."""
        file_state = {}
        for filepath in Path(".").glob("**/*.py"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    file_state[str(filepath)] = f.read()
            except Exception as e:
                logger.error(f"Error reading file {filepath}: {e}")
        return file_state

    def generate_diff(self, old_state, new_state):
        """Generate diff between two file states."""
        diff_output = ""

        all_files = set(old_state.keys()) | set(new_state.keys())

        for filepath in sorted(all_files):
            old_content = old_state.get(filepath, "")
            new_content = new_state.get(filepath, "")

            if old_content != new_content:
                diff = ""
                if self.repo:
                    # If in a Git repo, use git diff
                    try:
                        old_blob = self.repo.git.hash_object(old_content, '-w')
                        new_blob = self.repo.git.hash_object(new_content, '-w')
                        diff = self.repo.git.diff(old_blob, new_blob, unified=3)
                    except Exception as e:
                        logger.error(f"Error generating Git diff for {filepath}: {e}")

                if not diff:
                    # Fallback to basic diff if not in a Git repo or Git diff fails
                    diff = f"--- a/{filepath}\n+++ b/{filepath}\n"
                    diff += "".join(f"-{line}" if i < len(old_lines) and (j >= len(new_lines) or old_lines[i] != new_lines[j]) else f"+{line}" if j < len(new_lines) else "" for i, j, old_lines, new_lines in [(i, j, old_content.splitlines(keepends=True), new_content.splitlines(keepends=True)) for i in range(len(old_lines)) for j in range(len(new_lines))])

                diff_output += diff + "\n"

        return diff_output
    
    def add_code_version(self, code):
        """Adds a new code version to the dictionary, escaping HTML characters."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        escaped_code = html.escape(code)  # Escape HTML characters
        self.code_versions[timestamp] = escaped_code

    def get_code_versions_html(self):
        """Returns the code versions as HTML for display."""
        html_output = ""
        for timestamp, code in self.code_versions.items():
            html_output += f"<p><b>Version: {timestamp}</b></p>"
            html_output += f"<pre><code>{code}</code></pre><hr>"
        return html_output

class LLMManager:
    def __init__(self, versions):
        logger.info("Initializing LLMManager...")
        self.versions = versions
        try:
            self.llama_api = OpenAI(
                api_key="api_key",
                base_url="http://127.0.0.1:1234/v1/"
            )

            # Track execution context and test passes
            self.last_execution_locals = {}
            self.passed_tests_count = 0
            self.max_passed_tests = 2

            # Enhanced system message with code and test instructions
            self.system_message = {
                "role": "system",
                "content": """You are an AI assistant with Python code execution capabilities.

1. For code execution, use:
RUN-CODE
```python
your_code_here
Use code with caution.
Python
For tests, use:
TEST-ASSERT

assert condition, "Test message"
Use code with caution.
Python
Important rules:

Each block must start with its marker on its own line

Code must be within triple backticks with 'python' specified

Tests have access to variables from code execution

Generation stops after 2 successful test passes

Example:
I'll create a function and test it.

RUN-CODE

def add(a, b):
    return a + b
result = add(5, 7)
print(f'Result: {result}')
Use code with caution.
Python
TEST-ASSERT

assert result == 12, "Addition should work"
assert add(-1, 1) == 0, "Should handle negatives"
```"""
            }

            self.model_a_id = "qwen2.5-coder-7b-instruct"
            self.model_b_id = "qwen2.5-coder-14b-instruct"
            self.conversation = [self.system_message]

        except Exception as e:
            logger.error(f"Failed to initialize LLMManager: {e}")
            raise

    def run_code(self, code):
        """Execute code with safety checks and diff tracking."""
        logger.info("Preparing to execute code block with diff tracking")
        logger.debug(f"Code to execute:\n{code}")

        # Capture file state before execution
        old_state = self.versions.capture_file_state()

        # Basic safety checks
        dangerous_patterns = [
            "rm -rf",
            "system(",
            "eval(",
            "exec(",
            "input(",
            "requests.",
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                error_msg = f"Potentially unsafe code detected: {pattern}"
                logger.warning(error_msg)
                return error_msg

        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Set up safe execution environment
            safe_globals = {
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
                'os': os,
                'Path': Path
            }

            # Execute code and save locals for test access
            self.last_execution_locals = {}
            exec(code, safe_globals, self.last_execution_locals)
            output = captured_output.getvalue()
            logger.info("Code execution successful")

            # Capture file state after execution
            new_state = self.versions.capture_file_state()

            # Generate diff
            diff = self.versions.generate_diff(old_state, new_state)
            if diff:
                logger.info("Diff generated")
                logger.debug(f"Generated diff:\n{diff}")

            # Append the copy/paste footer to the output
            output += "\n\n---\nHave fun y'all! 🤠🪄🤖\n"

            return output
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Code execution failed: {str(e)}\n{error_trace}")
            return f"Error executing code:\n{error_trace}"
        finally:
            sys.stdout = old_stdout

    def run_tests(self, test_code):
        """Execute test assertions with access to previous code context."""
        logger.info("Running test assertions")
        logger.debug(f"Test code:\n{test_code}")

        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            # Include previous execution context in test environment
            test_globals = {
                'print': print,
                'assert': assert_,
                **self.last_execution_locals
            }

            exec(test_code, test_globals, {})
            output = captured_output.getvalue()
            self.passed_tests_count += 1
            logger.info(f"Tests passed. Count: {self.passed_tests_count}")
            
            # Add the code to the versions dictionary after tests pass
            self.versions.add_code_version(test_code)
            
            return "unit tests passed"
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

    def query_llama(self, model, messages, stream=False):
        """Query the LLM model with streaming support."""
        logger.info(f"Querying model: {model}")

        try:
            chat_completion = self.llama_api.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=stream,
                top_p=0.95,
                presence_penalty=0,
                frequency_penalty=0
            )

            if stream:
                try:
                    for chunk in chat_completion:
                        content = None

                        if hasattr(chunk.choices[0], 'delta'):
                            content = getattr(chunk.choices[0].delta, 'content', None)
                        elif hasattr(chunk.choices[0], 'text'):
                            content = chunk.choices[0].text
                        elif hasattr(chunk.choices[0], 'message'):
                            content = chunk.choices[0].message.content

                        if content:
                            yield content

                except Exception as e:
                    error_msg = f"Error in stream: {str(e)}"
                    logger.error(error_msg)
                    yield f"Error: {error_msg}"
            else:
                return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            error_msg = f"Error querying model {model}: {str(e)}"
            logger.error(error_msg)
            if stream:
                yield f"Error: {error_msg}"
            else:
                return f"Error: {error_msg}"

    def process_message(self, message):
        """Process a user message with code execution and testing."""
        logger.info("Processing new user message")
        self.passed_tests_count = 0  # Reset test counter

        if not message.strip():
            logger.warning("Empty message received")
            yield "Please enter a message", "Empty message received"
            return

        try:
            self.conversation.append({"role": "user", "content": message})

            # Process Model A
            logger.info("Getting Model A response")
            response_a = ""

            try:
                async_response = self.query_llama(self.model_a_id, self.conversation, stream=True)
                for chunk in async_response:
                    if isinstance(chunk, str) and chunk.startswith("Error"):
                        yield chunk, "Error occurred with Model A"
                        return
                    response_a += chunk
                    yield f"Model A Response:\n{response_a}\n\nModel B Response: Waiting...", "Processing Model A response..."

                    if self.should_stop_generation():
                        logger.info("Stopping generation - required test passes achieved")
                        yield f"Model A Response:\n{response_a}\n\nGeneration stopped: Required test passes achieved", "Complete"
                        return

            except Exception as e:
                error_msg = f"Error getting Model A response: {str(e)}"
                logger.error(error_msg)
                yield error_msg, "Error with Model A"
                return

            self.conversation.append({"role": "assistant", "name": self.model_a_id, "content": response_a})

            # Process code and test blocks from Model A
            code_blocks = re.findall(r'RUN-CODE\n```(?:python)?\n(.*?)\n```', response_a, re.DOTALL)
            test_blocks = re.findall(r'TEST-ASSERT\n```python\n(.*?)\n```', response_a, re.DOTALL)

            if code_blocks:
                logger.info(f"Found {len(code_blocks)} code block(s) in Model A response")
                for i, code in enumerate(code_blocks, 1):
                    logger.info(f"Executing code block {i}")
                    output = self.run_code(code.strip())

                    # Run associated tests if they exist
                    if i <= len(test_blocks):
                        test_result = self.run_tests(test_blocks[i - 1].strip())
                        output += f"\n{test_result}"

                    code_response = f"Code block {i} output:\n{output}"
                    self.conversation.append({"role": "assistant", "name": self.model_a_id, "content": code_response})
                    yield f"Model A Response:\n{response_a}\n\nCode Output:\n{output}\n\nModel B Response: Waiting...", f"Executed code block {i} from Model A"

                    if self.should_stop_generation():
                        logger.info("Stopping generation - required test passes achieved")
                        yield f"Model A Response:\n{response_a}\n\nGeneration stopped: Required test passes achieved", "Complete"
                        return

            # Process Model B if needed
            if not self.should_stop_generation():
                logger.info("Getting Model B response")
                response_b = ""
                try:
                    async_response = self.query_llama(self.model_b_id, self.conversation, stream=True)
                    for chunk in async_response:
                        if isinstance(chunk, str) and chunk.startswith("Error"):
                            yield f"Model A Response:\n{response_a}\n\nModel B Response: Error occurred", "Error occurred with Model B"
                            return
                        response_b += chunk
                        yield f"Model A Response:\n{response_a}\n\nModel B Response:\n{response_b}", "Processing Model B response..."

                        if self.should_stop_generation():
                            logger.info("Stopping generation - required test passes achieved")
                            yield f"Model A Response:\n{response_a}\n\nModel B Response:\n{response_b}\n\nGeneration stopped: Required test passes achieved", "Complete"
                            return

                except Exception as e:
                    error_msg = f"Error getting Model B response: {str(e)}"
                    logger.error(error_msg)
                    yield f"Model A Response:\n{response_a}\n\nModel B Response: Error: {error_msg}", "Error with Model B"
                    return

                self.conversation.append({"role": "assistant", "name": self.model_b_id, "content": response_b})

                # Process code and test blocks from Model B
                code_blocks = re.findall(r'RUN-CODE\n```(?:python)?\n(.*?)\n```', response_b, re.DOTALL)
                test_blocks = re.findall(r'TEST-ASSERT\n```python\n(.*?)\n```', response_b, re.DOTALL)

                if code_blocks:
                    for i, code in enumerate(code_blocks, 1):
                        output = self.run_code(code.strip())

                        if i <= len(test_blocks):
                            test_result = self.run_tests(test_blocks[i - 1].strip())
                            output += f"\n{test_result}"

                        code_response = f"Code block {i} output:\n{output}"
                        self.conversation.append({"role": "assistant", "name": self.model_b_id, "content": code_response})
                        yield f"Model A Response:\n{response_a}\n\nModel B Response:\n{response_b}\n\nCode Output:\n{output}", f"Executed code block {i} from Model B"

                        if self.should_stop_generation():
                            logger.info("Stopping generation - required test passes achieved")
                            yield f"Model A Response:\n{response_a}\n\nModel B Response:\n{response_b}\n\nGeneration stopped: Required test passes achieved", "Complete"
                            return

                yield f"Model A Response:\n{response_a}\n\nModel B Response:\n{response_b}", "Completed"

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            yield error_msg

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
            system_message = self.conversation[0]  # Save system message
            self.conversation = [system_message]  # Reset with only system message
            self.passed_tests_count = 0  # Reset test counter
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
        versions = Versions()
        manager = LLMManager(versions)

        with gr.Blocks(title="LLM Pattern Interface") as interface:
            gr.Markdown("# 🚂🤖🪄 Conductor")
            gr.Markdown("Enter your message to interact with the AI models. Code will be executed and tested until pass criteria are met.")

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

            with gr.Column(scale=3):
                code_versions_display = gr.HTML(
                    label="Generated Code Versions"
                )

            with gr.Row():
                show_versions_btn = gr.Button("Show Code Versions")
                clear_versions_btn = gr.Button("Clear Versions")

            status_display = gr.Textbox(
                label="Status/Tests",
                lines=2,
                interactive=False,
                visible=True
            )

            def handle_submit(message):
                """Handle message submission with streaming."""
                if not message:
                    return "", "Please enter a message"

                try:
                    logger.info(f"Handling new message: {message[:50]}...")
                    manager.should_stop = False

                    message_generator = manager.process_message(message)

                    for result in message_generator:
                        if isinstance(result, tuple):
                            yield result[0], result[1], versions.get_code_versions_html()
                        else:
                            yield result, "Processing...", versions.get_code_versions_html()

                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    logger.error(error_msg)
                    yield "", error_msg, versions.get_code_versions_html()

            def handle_stop():
                """Handle stop button click."""
                manager.should_stop = True
                return "Stopping generation...", "Stopping..."

            def handle_clear():
                """Handle conversation clearing."""
                try:
                    result = manager.clear_conversation()
                    return "", result
                except Exception as e:
                    error_msg = f"Error clearing conversation: {str(e)}"
                    logger.error(error_msg)
                    return "", error_msg

            def handle_show_versions():
                """Handle show code versions button click."""
                return versions.get_code_versions_html()

            def handle_clear_versions():
                """Handle clear versions button click."""
                versions.code_versions = {}
                return ""

            # Wire up the interface events
            submit_btn.click(
                fn=handle_submit,
                inputs=input_message,
                outputs=[conversation_display, status_display, code_versions_display],
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
                outputs=[conversation_display, status_display]
            )

            # Wire up the code versions-related events
            show_versions_btn.click(
                fn=handle_show_versions,
                inputs=None,
                outputs=code_versions_display
            )

            clear_versions_btn.click(
                fn=handle_clear_versions,
                inputs=None,
                outputs=code_versions_display
            )

            # Show conversation history on load
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
    logger.info("Starting LLM Interface application")

    try:
        # Ensure we're running in a virtual environment
        restart_in_venv()

        # Create and launch the interface
        interface = create_ui()
        logger.info("Launching Gradio interface")
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=4321,
            debug=True
        )

    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()
