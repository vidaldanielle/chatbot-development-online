from pypdf import PdfReader

reader = PdfReader(
    "data/test_company_handbook.pdf"
)

print("Pages:", len(reader.pages))

for page in reader.pages:
    print(page.extract_text())