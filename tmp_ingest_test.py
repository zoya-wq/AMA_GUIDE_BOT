from ingestion_pipeline import AMAGuidesIngestionPipeline
from pathlib import Path
import asyncio

pdf = Path('C:/Users/ASUS/Documents/AF/AMA_GUIDE/AMA GUIDE 5TH - COMPLETE (1).pdf')
print('PDF exists', pdf.exists())

pipeline = AMAGuidesIngestionPipeline()
print('pipeline created')

with open(pdf, 'rb') as f:
    pdf_bytes = f.read()

try:
    progress = asyncio.run(pipeline.run_pipeline(pdf_bytes))
    print('Done', progress.processed_pages, 'pages')
    print('errors', progress.errors)
except Exception as e:
    print('Exception', type(e), e)
