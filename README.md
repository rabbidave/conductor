# 🚂🤖🪄Conductor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* Deployable interface (GUI/API) that (locally) orchestrates multiple (remote and/or local) Language Models.

* Python code is executed and tested against assertions; via configurable allow/block list of operations (regex).

![](https://github.com/rabbidave/conductor/blob/main/envvar.gif?raw=true)

## Features

*   **Multi LM Interaction:** Engage with multiple LMs sequentially for mob-style programming.
*   **Safe Code Execution:** Run Python Code within a sandboxed environment via allowed/restricted operations.
*   **Automated Test Assertions:** Define test cases using `TEST-ASSERT` blocks; Conductor runs them automatically.
*   **Test-Driven Generation:** Generation stops after a configurable number of successful test passes; TDD for AI
*   **Customizable System Message:** Tailor the behavior of the LMs by modifying the system message.
*   **Configurable Model IDs, API URLs, and Generation Parameters:** Easily switch between different LMs and Generation Settings
*   **Detailed Logging:** Comprehensive logs available in the `logs/` directory for debugging.

## Prerequisites

*   **Python 3.7+:** Ensure you have Python 3.7 or a newer version installed.
*   **LM Studio (or similar):** A local LLM server like LM Studio is required. You can download it from [LM Studio's website](https://lmstudio.ai/).

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/rabbidave/conductor
    cd conductor
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This will install `gradio`, `openai`, and `GitPython`.

## Configuration

1.  **LM Studio Setup:**

    *   Launch LM Studio.
    *   Download and load the models you want to use (e.g., `exaone-3.5-32b-instruct@q4_k_m` and `qwq-32b-preview`).
    *   Start the local server in LM Studio (default: `http://localhost:1234`).

2.  **Environment Variables:**

    *   On the first use of the project, the system will automatically create an `.env` file with the default values. You can modify this file to change the local server URL, model IDs, and other parameters.
    *   Locate the `.env` file in the project root folder.
    *   Update the following variables with the correct values:
        *   `MODEL_A_ID`: Model ID for the first LLM (e.g., `"exaone-3.5-32b-instruct@q4_k_m"`).
        *   `MODEL_B_ID`: Model ID for the second LLM (e.g., `"qwq-32b-preview"`).
        *   `MODEL_A_URL`: API URL for Model A (e.g., `"http://localhost:1234/v1/"`).
        *  `MODEL_B_URL`: API URL for Model B (e.g., `"http://localhost:1235/v1/"`).
        *   `MAX_TOKENS`: Maximum number of tokens to generate (e.g., `"2000"`).
        *   `TEMPERATURE`: Sampling temperature for generation (e.g., `"0.7"`).
        *  `TOP_P`: Top-p value for nucleus sampling (e.g., `"0.95"`).

    ```
    MODEL_A_ID="your-model-a-id"  # e.g., "exaone-3.5-32b-instruct@q4_k_m"
    MODEL_B_ID="your-model-b-id"  # e.g., "qwq-32b-preview"
    MODEL_A_URL="http://localhost:1234/v1/"
    MODEL_B_URL="http://localhost:1235/v1/"
    MAX_TOKENS="2000"
    TEMPERATURE="0.7"
    TOP_P="0.95"
    ```
3.  **UI-Based Configuration:**

    *   You can also modify the above environment variables directly within the Conductor UI.
    *   Expand the "Environment Variables" accordion.
    *   Enter the desired values and click "Update Environment Variables".
    *   **Note:** Changes made via the UI will update the environment variables but will require an application restart for them to fully take effect.

4.  **System Message (Optional):**

    *   In `LLMManager.__init__`, you can customize the `self.system_message` to modify the behavior of the LMs. This message sets the context for the conversation and defines the `RUN-CODE` and `TEST-ASSERT` block formats.

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
    
5.  **Test Pass Count:**

    *   To change the number of successful test passes required to stop generation, modify `self.max_passed_tests` in `LLMManager.__init__` in the main python file.

6.  **Logging:**

    *   Logs are stored in the `logs/` directory.
    *   Adjust the logging level in the logging setup section of `app.py` using:

    ```python
    logging.basicConfig(level=logging.DEBUG)  # For verbose logging
    # or
    logging.basicConfig(level=logging.INFO)   # For less verbose logging
    ```
    *   Debug level logging can be enabled to provide very detailed output on the behaviour of the code.

7. **Security Note**
 * Always use caution with code generated by an LLM and do not execute code from untrusted sources.

## Usage

1.  **Start the Interface:**

    ```bash
    python app.py
    ```

2.  **Access the UI:**

    *   Open your web browser and go to `http://localhost:1339` (or the address indicated in the terminal).

3.  **Interact with the LMs:**

    *   Enter your prompt in the "Input Message" textbox.
    *   Click "Submit" to send the message to Model A.
    *   The conversation and results will appear in the "Conversation & Results" textbox.
    *   Code execution and test results will also be displayed.
    *   If the required number of tests pass, generation will stop. Otherwise, Model B will be engaged.
    *   Click "Stop Generation" to manually stop the generation process.
    *   Click "Clear Conversation" to start a new conversation.
    *   Use the "Show Last Code" and "Show Last Output" buttons to view the last executed code and output.

## Example Workflow

1.  **Prompt:**

    ```
    Write a Python function to calculate the nth term of the Fibonacci sequence using recursion. Include tests to validate the results for n=0, n=1, n=5, and n=10.
    ```

2.  **Model A's Response (may vary):**

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

3.  **Execution and Testing:**

    *   Conductor will execute the `RUN-CODE` block.
    *   It will then run the `TEST-ASSERT` block.
    *   The results (output and test outcomes) will be displayed.
    *   The output in the UI will include the footer:

        ```
        Code block 1 output:

        ---
        Have fun y'all! 🤠🪄🤖
        ```

## Troubleshooting

*   **Package Installation Errors:** If you encounter errors during package installation, ensure your virtual environment is activated, and you have the necessary permissions to install packages.
*   **LM Studio Connection Issues:** Verify that LM Studio is running and the local server is started. Check the port number (default: 1234) and make sure it matches the `MODEL_A_URL` or `MODEL_B_URL` in the `.env` file, or the values in the "Environment Variables" section of the UI.
*   **Model Not Found:** Double-check that the model IDs you've configured in the `.env` file (or via the UI) are correct and that the models are loaded in LM Studio.
*    **Firewall Issues:** Ensure your firewall is not blocking requests to the models from the interface
*   **Git Errors:** If you get errors related to Git, make sure Git is installed and that the project is inside a Git repository. If you don't want to use Git, the code will fall back to skipping diff generation.
* **Terminal Output:** If you're not seeing output in the terminal, ensure you are running the application with an active virtual environment and the environment encoding is set to UTF-8.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please adhere to the following guidelines:

*   **Bug Reports:** Submit detailed bug reports including the steps to reproduce and any relevant error messages.
*   **Feature Requests:** Suggest new features, enhancements, or improvements in an issue.
*   **Pull Requests:** When submitting pull requests, make sure that your code aligns with the project's style and standards. Add tests where necessary and make sure all tests pass before submitting.

Please feel free to open issues or submit pull requests on the project's repository page.
content_copy
Use code with caution.
Markdown

Key Updates:

Environment Variable Explanation: Clarified the use of the .env file and included all the relevant environment variables for configuration, including the new parameters like MAX_TOKENS, TEMPERATURE and TOP_P.

UI-Based Configuration: Added a section to describe how to configure environment variables within the UI.

Dynamic Configuration: Included a mention of being able to dynamically configure the API/Model URLs and IDs via UI.

Troubleshooting: Added firewall issues to the troubleshooting section.

Terminal Output Troubleshooting: Added a troubleshooting point about terminal output.

UI Elements: Added the ability to show the last code and output

Port update: Updated the default port to 1339.

This revised README.md should provide a much clearer and more comprehensive guide for users of your project.
