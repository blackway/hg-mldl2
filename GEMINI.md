# GEMINI.md

## Project Overview

This repository contains the code examples for the book "<혼자 공부하는 머신러닝+딥러닝(개정판)>" (Self-study machine learning + deep learning (revised edition)). It's a collection of Jupyter notebooks that cover various machine learning and deep learning topics. The notebooks provide hands-on examples using popular Python libraries.

The main technologies used in this project are:
*   **Python**: The primary programming language.
*   **Jupyter Notebook**: For interactive coding and explanations.
*   **Scikit-learn**: For classical machine learning algorithms.
*   **Keras**: As a high-level API for building and training deep learning models.
*   **PyTorch**: An open-source machine learning framework.

## Directory Overview

The directory is structured around the chapters of the book. Each `.ipynb` file corresponds to a specific section of the book. Some chapters have separate notebooks for Keras and PyTorch implementations, for instance:
*   `07-2.ipynb`: Keras implementation.
*   `07-2.pytorch.ipynb`: PyTorch implementation.

The repository also contains saved models like `best-cnn-model.pt`, `best-2lstm-model.keras`, etc.

## Key Files

*   **`*.ipynb`**: Jupyter notebooks containing the code examples. These are the core of the repository.
*   **`README.md`**: Provides a general introduction to the repository.
*   **`requirements.txt`**: Lists the Python dependencies for this project.
*   **`pyproject.toml`**: Basic project configuration.
*   **`best-*.keras`, `best-*.pt`**: Saved model files.

## Usage

To run the code in this repository, you will need to have Python and the required libraries installed. You can install the libraries using pip:

```bash
pip install -r requirements.txt
```

Then, you can run Jupyter lab to open and run the notebooks:

```bash
jupyter lab
```

This will open a new tab in your browser where you can navigate to the notebook files and execute the code cells.
