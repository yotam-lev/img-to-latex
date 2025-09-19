Image Processing and Symbol Recognition Application

Overview
This project is an image processing and symbol recognition application built with Python. It processes uploaded images, detects symbols (including fractions), and converts them into LaTeX code. The application includes a graphical user interface for user interaction.

Features
The application allows users to upload images, detect bounding boxes for symbols, identify fractions, classify mathematical symbols, and generate LaTeX code. It optionally displays intermediate processing steps for analysis.

Recognized Symbols
The system currently recognizes the following symbols: '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'X', 'alpha', 'beta', 'geq', 'gt', 'lambda', 'leq', 'lt', 'neq', and 'y'.

Project Structure
The project consists of several Python files. The display_ui.py script manages the graphical user interface. The image_processing.py script handles image manipulation, such as detecting and merging bounding boxes. The main.py file serves as the entry point for the application and integrates all components. The predict_symbol.py script contains the neural network model for symbol recognition, and the single_symbol_processing.py script handles preprocessing of individual symbols.

There are also Jupyter Notebook files included in the project. These notebooks contain parts of the code that are implemented in the Python files. The model training notebook demonstrates the training process for the recognition model. The generator.ipynb file was used to generate the testing dataset.

Prerequisites
Python 3.8 or higher is required. The following Python packages need to be installed: numpy, opencv-python, torch, torchvision, matplotlib, Pillow, scikit-image, and tkinter.

Setup Instructions
To set up the project, clone the repository, install the dependencies, and download the pre-trained model file named best_model.pth. Place this file in the project root directory. The application must be run from the main.py file. Use the command python main.py to launch the application.

How It Works
The graphical user interface allows users to upload images. The system processes the image to detect bounding boxes, identify fractions, and classify symbols. It outputs the recognized symbols as LaTeX code. Users can enable an option to visualize intermediate processing steps.

Limitations and Future Work
The application is currently limited to recognizing predefined symbols. Future improvements could include support for more complex mathematical expressions, better OCR capabilities for handwritten symbols, and deployment as a web-based application.

License
This project is licensed under the MIT License.