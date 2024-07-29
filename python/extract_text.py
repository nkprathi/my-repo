import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  
                text += page_text
    return text

def save_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

pdf_path = '/Users/prathi/work/input/Singapore.pdf'
output_file_path = '/Users/prathi/work/output/s_extracted_text.txt'

# Extract text from the PDF
extracted_text = extract_text_from_pdf(pdf_path)

# Save the extracted text to a file
save_text_to_file(extracted_text, output_file_path)

print(f"Text extracted and saved to {output_file_path}")



# import pdfplumber

# def extract_text_from_pdf(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         text = ''
#         for page in pdf.pages:
#             text += page.extract_text()
#         return text

# pdf_path = '/Users/prathi/work/my-repo/pdf_data_folder/png2pdf.pdf'
# text = extract_text_from_pdf(pdf_path)
# print(text)