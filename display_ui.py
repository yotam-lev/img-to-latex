import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
from matplotlib.figure import Figure
import matplotlib
import image_processing
import predict_symbol
import single_symbol_processing

# Use a non-interactive backend for Tkinter
matplotlib.use("TkAgg")


def upload_image(image_label, show_process_var, latex_text_field):
    def inner():
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        if file_path:
            # Load and display the image
            img = Image.open(file_path)
            img.thumbnail((300, 300))  # Resize the image to fit in the window
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk  # Keep a reference to prevent garbage collection

            # Run the script on the uploaded image
            run_script(file_path, show_process_var, latex_text_field)

    return inner


def run_script(image_path, show_process_var, latex_text_field):
    try:
        # Determine whether to show intermediate steps
        show_image = show_process_var.get()

        # Load the uploaded image
        pil_image = Image.open(image_path)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Load the model
        model = predict_symbol.SimpleCNN(num_classes=23)
        model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
        model.eval()

        # Run the processing script
        fig = Figure(figsize=(8, 8))  # Create a figure for displaying plots

        img_with_boxes, boxes = image_processing.detect_boxes(image, show_image=show_image)
        img_with_fractions, fraction_bars = image_processing.detect_fraction_bars(
            image, boxes, show_image=show_image
        )
        latex_code = ""

        if fraction_bars:
            sections, numerator_indices = image_processing.extract_fractions(
                img=image,
                boxes=boxes,
                fraction_bar_boxes=fraction_bars,
                show_images=show_image
            )

            for i, section in enumerate(sections):
                section_img_with_boxes, section_boxes = image_processing.detect_boxes(
                    section['region'], show_image=show_image
                )
                section_img_merged_boxes, section_boxes = image_processing.merge_boxes(
                    section['region'], section_boxes, show_image=show_image
                )
                box_images = image_processing.extract_box_images(
                    section['region'], section_boxes, show_images=show_image
                )

                processed_images = []
                for img in box_images:
                    processed_images.append(
                        single_symbol_processing.process_image(
                            img, show_process=show_image, show_image=show_image
                        )
                    )

                predictions = predict_symbol.predict_images_pytorch(processed_images, "best_model.pth")

                if section['type'] == 'numerator':
                    latex_code += "\\frac{" + "".join(
                        [
                            f"\\{pred['class_name']}" if pred['class_name'] in [
                                'alpha', 'beta', 'lambda', 'geq', 'leq', 'neq', 'lt', 'gt'
                            ] else pred['class_name']
                            for pred in predictions
                        ]
                    ) + "}{"
                elif section['type'] == 'denominator':
                    latex_code += "".join(
                        [
                            f"\\{pred['class_name']}" if pred['class_name'] in [
                                'alpha', 'beta', 'lambda', 'geq', 'leq', 'neq', 'lt', 'gt'
                            ] else pred['class_name']
                            for pred in predictions
                        ]
                    ) + "}"
                else:
                    latex_code += "".join(
                        [
                            f"\\{pred['class_name']}" if pred['class_name'] in [
                                'alpha', 'beta', 'lambda', 'geq', 'leq', 'neq', 'lt', 'gt'
                            ] else pred['class_name']
                            for pred in predictions
                        ]
                    )
        else:
            merged_img, merged_boxes = image_processing.merge_boxes(image, boxes, show_image=show_image)
            box_images = image_processing.extract_box_images(image, merged_boxes, show_images=show_image)

            processed_images = []
            for img in box_images:
                processed_images.append(
                    single_symbol_processing.process_image(img, show_process=show_image, show_image=show_image)
                )

            predictions = predict_symbol.predict_images_pytorch(processed_images, "best_model.pth")
            latex_code += "".join(
                [
                    f"\\{pred['class_name']}" if pred['class_name'] in [
                        'alpha', 'beta', 'lambda', 'geq', 'leq', 'neq', 'lt', 'gt'
                    ] else pred['class_name']
                    for pred in predictions
                ]
            )

        # Display the LaTeX code in the text field
        latex_text_field.delete("1.0", tk.END)
        latex_text_field.insert(tk.END, latex_code)

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")


def main():
    # Set up the main application window
    app = tk.Tk()
    app.title("Image Uploader")
    app.geometry("500x600")  # Set default window size

    # Set the app icon
    try:
        app.iconphoto(False, tk.PhotoImage(file="icon.png"))
    except Exception as e:
        print("Error loading icon:", e)

    # Add a label
    label = tk.Label(app, text="Upload an image")
    label.pack(pady=10)

    # Add an upload button
    image_label = tk.Label(app)
    show_process_var = tk.BooleanVar(value=False)
    latex_text_field = tk.Text(app, height=10, wrap=tk.WORD)

    upload_button = tk.Button(app, text="Upload Image", command=upload_image(image_label, show_process_var, latex_text_field))
    upload_button.pack(pady=5)

    # Add a checkbox to enable/disable showing intermediate steps
    show_process_checkbox = tk.Checkbutton(app, text="Show Intermediate Steps", variable=show_process_var)
    show_process_checkbox.pack(pady=5)

    # Add a label to display the uploaded image
    image_label.pack(pady=10)

    # Add a text field to display LaTeX code
    latex_text_field.pack(pady=10, fill=tk.BOTH, expand=True)

    # Start the application
    app.mainloop()
