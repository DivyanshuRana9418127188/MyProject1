import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file_object):
    """
    Extracts text from the given PDF file-like object.
    
    Args:
        pdf_file_object: A file-like object representing the PDF, 
                         typically obtained from st.file_uploader.
    
    Returns:
        str: The extracted text from the PDF.
    """
    # fitz.open can directly accept a file-like object (like those from st.file_uploader)
    doc = fitz.open(stream=pdf_file_object.read(), filetype="pdf") 
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    doc.close() # Good practice to close the document
    return text