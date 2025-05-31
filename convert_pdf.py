from paths import MISC_DIR
import os
from fpdf import FPDF

def text_to_pdf_with_images(text_file, image_paths, output_pdf):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=12)

    # Read and add text
    with open(text_file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Avoid adding blank lines
                pdf.cell(0, 5, line, ln=True)

    # Add images (each on a new page)
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        try:
            pdf.add_page()
            # Resize image to fit within page margins
            pdf.image(img_path, x=10, y=10, w=pdf.w - 20)
        except RuntimeError as e:
            print(f"Error loading image '{img_path}': {e}")
            continue

    # Save PDF
    pdf.output(output_pdf)
    print(f"PDF saved as: {output_pdf}")

# Example usage
text_file_path = f"{MISC_DIR}/hyperparams.txt"
image_list = [f"{MISC_DIR}/confusion_matrix.png", f"{MISC_DIR}/plot.png"]  # List of image paths
output_pdf_path = "output.pdf"

text_to_pdf_with_images(text_file_path, image_list, output_pdf_path)