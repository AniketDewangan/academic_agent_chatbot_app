# Attention

This is the code for all the open-source LLM models. It needs a little fine-tuning and a few adjustments, which is why I didn’t use it.

### To run this:
Run the following code in the terminal:

```bash
streamlit run Attention.py
```

# Attention_1

This is the code using the Gemini API key.

### To run this:
Run the following command in the terminal:

```bash
streamlit run Attention_1.py
```

## Installation

To set up the environment for this tool, first ensure you have Python installed. Then, install the required libraries using the `Requirements.txt` file provided.

```bash
pip install -r Requirements.txt
```

## Key Dependencies

Streamlit: Used for building an interactive web interface.

Transformers & Sentence-Transformers: For leveraging transformer models.

Faiss-cpu: For efficient similarity search.

Pdfplumber: Helps in parsing and extracting data from PDFs.

Torch: Powers deep learning model operations.

Google Generative AI: Optional module for AI-enhanced features.


For a complete list of dependencies, please refer to the Requirements.txt file.

## Usage

1. Launching the Interface: Run the following command to launch the Streamlit interface:

```bash
streamlit run Attention.py
```


3. Dataset and Model Selection: Use the interface to select datasets, models, and other parameters for analysis. The tool will guide you through the choices and configuration.


4. Analysis: The tool automatically extracts features, preprocesses data, and runs classification models based on your configuration. It provides comparisons and accuracy metrics across models and datasets.



## Contributing

If you would like to contribute to the project, feel free to open a pull request. We welcome improvements and suggestions, especially those enhancing the dataset, model comparison, and performance analysis features.

## License

This project is licensed under the MIT License.

This code will display as a well-structured README on platforms like GitHub. Let me know if you need further customization!


