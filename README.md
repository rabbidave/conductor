# 🚂🤖🪄Conductor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Conductor is an interactive interface that (locally) orchestrates multiple (remote or local) Language Models with Python code execution and test assertion capabilities. 

It features a dedicated code versioning system that displays a scrollable view of generated and tested code, along with Aider-style Git diffs (or basic line-by-line diffs if Git is unavailable) to track code changes.

## Features

*   **Dual LLM Interaction:** Engage with multiple LMs sequentially for mob-style programming.
*   **Safe Code Execution:** Run Python Code within a sandboxed environment via allowed/restricted operations.
*   **Automated Test Assertions:** Define test cases using `TEST-ASSERT` blocks; Conductor will automatically run them against the executed code.
*   **Test-Driven Generation:** Generation stops after a configurable number of successful test passes, encouraging test-driven development.
*   **Code Versioning:**  Tested code is stored and displayed in a scrollable HTML view within the UI, providing a clear history of generated code.
*   **Aider-style Git Diffs:** Track code changes with Git diffs (or a basic line-by-line diff if not in a Git repository), making it easy to review and integrate code into your projects. (Note: Diffs are displayed temporarily during code execution).
*   **Real-time Streaming:** Responses from the LLMs are streamed in real-time to the user interface.
*   **Conversation History:** View the entire conversation history with the LMs.
*   **Status and Error Display:** Clear status updates and error messages guide you through the interaction.
*   **Customizable System Message:** Tailor the behavior of the LMs by modifying the system message.
*   **Configurable Model IDs:** Easily switch between different local LLMs by updating the model IDs.
*   **Gradio-powered UI:** A user-friendly web interface powered by Gradio.
*   **Enhanced Copy-Paste:** Executed code blocks in the UI are appended with a footer, making it easier to copy and paste the generated code.

## Prerequisites

*   **Python 3.7+:** Ensure you have Python 3.7 or a newer version installed.
*   **LM Studio (or similar):** A local LLM server like LM Studio is required. You can download it from [LM Studio's website](https://lmstudio.ai/).
*   **Git (optional):** For the best diff tracking experience, Git should be installed and the project should be within a Git repository. If Git is not available, a basic line-by-line diff will be used.

## Installation

1. **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd conductor
    ```

2. **Create and Activate a Virtual Environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This will install `gradio`, `openai`, and `GitPython`.

## Configuration

1. **LM Studio Setup:**

    *   Launch LM Studio.
    *   Download and load the models you want to use (e.g., `exaone-3.5-32b-instruct@q4_k_m` and `qwq-32b-preview`).
    *   Start the local server in LM Studio (default: `http://localhost:1234`).

2. **Model IDs:**

    *   Open `gradio-llm-interface.py` in a text editor.
    *   Locate the `LLMManager.__init__` method.
    *   Update the `self.model_a_id` and `self.model_b_id` variables with the correct model IDs from LM Studio.

    ```python
    self.model_a_id = "your-model-a-id"  # e.g., "exaone-3.5-32b-instruct@q4_k_m"
    self.model_b_id = "your-model-b-id"  # e.g., "qwq-32b-preview"
    ```

3. **System Message (Optional):**

    *   In `LLMManager.__init__`, you can customize the `self.system_message` to modify the behavior of the LLMs. This message sets the context for the conversation and defines the `RUN-CODE` and `TEST-ASSERT` block formats.

    ```python
    self.system_message = {
        "role": "system",
        "content": """You are an AI assistant with Python code execution capabilities.

    1. For code execution, use:
    RUN-CODE
    ```python
    your_code_here
    ```

    2. For tests, use:
    TEST-ASSERT
    ```python
    assert condition, "Test message"
    ```

    3. Important rules:
    - Each block must start with its marker on its own line
    - Code must be within triple backticks with 'python' specified
    - Tests have access to variables from code execution
    - Generation stops after 2 successful test passes

    ... (rest of the system message)
    """
    }
    ```

4. **Test Pass Count:**

    *   To change the number of successful test passes required to stop generation, modify `self.max_passed_tests` in `LLMManager.__init__`.

5. **Logging:**

    *   Logs are stored in the `logs/` directory; [Observers coming soon](https://github.com/cfahlgren1/observers)
    *   Adjust the logging level in `gradio-llm-interface.py` using:

    ```python
    logging.basicConfig(level=logging.DEBUG)  # For verbose logging
    # or
    logging.basicConfig(level=logging.INFO)   # For less verbose logging
    ```

## Usage

1. **Start the Interface:**

    ```bash
    python gradio-llm-interface.py
    ```

2. **Access the UI:**

    *   Open your web browser and go to `http://localhost:1337` (or the address indicated in the terminal).

3. **Interact with the LLMs:**

    *   Enter your prompt in the "Input Message" textbox.
    *   Click "Submit" to send the message to Model A.
    *   The conversation and results will appear in the "Conversation & Results" textbox.
    *   Code execution and test results will also be displayed.
    *   If the required number of tests pass, generation will stop. Otherwise, Model B will be engaged.
    *   Click "Stop Generation" to manually stop the generation process.
    *   Click "Clear Conversation" to start a new conversation.

4. **View Code Versions:**

    *   After code has been executed and tests have passed, click the "Show Code Versions" button to view a scrollable HTML display of the generated code in the "Generated Code Versions" section.
    *   Click "Clear Versions" to clear the code version history.

5. **Copy/Paste Generated Code:**
    *   Executed code blocks will have the following footer in the UI, making it easier to copy the code:

    ```
    ---
    Have fun y'all! 🤠🪄🤖
    ```

## Example Workflow

1. **Prompt:**

    ```
    Write a Python function to calculate the nth term of the Fibonacci sequence using recursion. Include tests to validate the results for n=0, n=1, n=5, and n=10.
    ```

2. **Model A's Response (may vary):**

    ```
    RUN-CODE
    ```python
    def fibonacci_recursive(n):
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
    ```

    TEST-ASSERT
    ```python
    assert fibonacci_recursive(0) == 0, "fibonacci_recursive(0) should be 0"
    assert fibonacci_recursive(1) == 1, "fibonacci_recursive(1) should be 1"
    assert fibonacci_recursive(5) == 5, "fibonacci_recursive(5) should be 5"
    assert fibonacci_recursive(10) == 55, "fibonacci_recursive(10) should be 55"
    ```
    ```

3. **Execution and Testing:**

    *   Conductor will execute the `RUN-CODE` block.
    *   It will then run the `TEST-ASSERT` block.
    *   The results (output and test outcomes) will be displayed.
    *   The output in the UI will include the footer:

        ```
        Code block 1 output:

        ---
        Have fun y'all! 🤠🪄🤖
        ```

    *   If tests pass, the code will be added to the code versions.

4. **Show Code Versions:**
    *   Clicking "Show Code Versions" will display the code in the "Generated Code Versions" section:

    ```html
    <p><b>Version: 2023-10-27 10:30:00</b></p>
    <pre><code>def fibonacci_recursive(n):
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
    </code></pre>
    <hr>
    <p><b>Version: 2023-10-27 10:30:15</b></p>
    <pre><code>assert fibonacci_recursive(0) == 0, "fibonacci_recursive(0) should be 0"
    assert fibonacci_recursive(1) == 1, "fibonacci_recursive(1) should be 1"
    assert fibonacci_recursive(5) == 5, "fibonacci_recursive(5) should be 5"
    assert fibonacci_recursive(10) == 55, "fibonacci_recursive(10) should be 55"
    </code></pre>
    <hr>
    ```

## Troubleshooting

*   **Package Installation Errors:** If you encounter errors during package installation, ensure your virtual environment is activated, and you have the necessary permissions to install packages.
*   **LM Studio Connection Issues:** Verify that LM Studio is running and the local server is started. Check the port number (default: 1234) and make sure it matches the `base_url` in `LLMManager.__init__`.
*   **Model Not Found:** Double-check that the model IDs you've configured in `LLMManager.__init__` are correct and that the models are loaded in LM Studio.
*   **Git Errors:** If you get errors related to Git, make sure Git is installed and that the project is inside a Git repository. If you don't want to use Git, the code will fall back to a basic line-by-line diff.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests on the project's repository page.
content_copy
Use code with caution.
Markdown

Key Changes:

Code Versioning: Added a section explaining the new code versioning feature and how to use the "Show Code Versions" and "Clear Versions" buttons.

Updated Example Workflow: Modified the example to demonstrate the code versioning feature and how the code appears in the "Generated Code Versions" section.

Minor Refinements: Improved the overall clarity and flow of the README.

Remember to:

Create a LICENSE file (with the MIT License content) in your repository.

Create a requirements.txt file with the following content:

gradio
openai
GitPython
content_copy
Use code with caution.

Replace <repository_url> with the actual URL of your repository.

This comprehensive README.md should be very helpful for users of your improved Conductor project! Let me know if you have any more questions.
