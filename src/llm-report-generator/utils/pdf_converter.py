import os
import fpdf


FONT_PATH = os.path.dirname(os.path.abspath(__file__)) + "/fonts/"

## Write multiple pages
def sections_to_pdf(sections, filename):
    pdf = fpdf.FPDF(orientation='P', unit='mm', format='A4')
    fontsize_pt = 12
    margin_bottom_mm = 10
    
    ## Write each section. Helper function always creates new page by default
    for section in sections:
        pdf = __str_to_pdf_helper(section, pdf, fontsize_pt, margin_bottom_mm)
    
    pdf.output(filename, "F").encode("utf-8")
    return



def str_to_txt(text, filename):
    ## Write to text file
    mode = "x" if not os.path.exists(filename) else "w"
    with open(filename, mode, encoding="utf-8") as f:
        f.write(text)


## Write string to PDF
def str_to_pdf(text, filename):
    A4_WIDTH_MM = 210
    PT_TO_MM = 0.35
    fontsize_pt = 12
    # fontsize_mm = fontsize_pt * PT_TO_MM
    margin_bottom_mm = 10
    # character_width_mm = 6.2 * PT_TO_MM
    # width_text = A4_WIDTH_MM / character_width_mm

    pdf = fpdf.FPDF(orientation='P', unit='mm', format='A4')
    pdf = __str_to_pdf_helper(text, pdf, fontsize_pt, margin_bottom_mm)

    pdf.output(filename, "F").encode("utf-8")


## Helper function for writing string to PDF
def __str_to_pdf_helper(text, pdf: fpdf.FPDF, fontsize_pt, margin_bottom_mm):
    pdf.set_auto_page_break(True, margin=margin_bottom_mm)
    pdf.add_page()
    pdf.add_font("Roboto", "", FONT_PATH + "Roboto-Regular.ttf", uni=True)
    pdf.add_font("Roboto", "B", FONT_PATH + "Roboto-Bold.ttf", uni=True)
    pdf.set_font(family='Roboto', size=fontsize_pt)
    
    splitted = text.split('\n')

    ## Write word for word
    words = []
    for para in splitted:
        words.extend(para.split())
        words.append("\n")

    start_bold, end_bold = False, False
    for word in words:
        if word == "\n":
            pdf.ln()
            continue
        if word.startswith("**"):
            word = word[2:]
            start_bold = True
        if word.endswith("**"):
            word = word[:-2]
            end_bold = True
        if start_bold:
            pdf.set_font(family="Roboto", style="B", size=fontsize_pt)
            start_bold = False
        pdf.write(h=5, txt=word+" ")
        if end_bold:
            pdf.set_font(family="Roboto", style="", size=fontsize_pt)
            end_bold = False

    return pdf


def txt_to_pdf(source_filename, output_filename):
    with open(source_filename, "r", encoding="utf-8") as f:
        text = f.read()
        str_to_pdf(text, output_filename)
