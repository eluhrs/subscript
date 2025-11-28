import os
import logging
from typing import List, Dict, Any
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
from modules.interfaces import OutputEngine

logger = logging.getLogger(__name__)

class UnifiedOutputEngine(OutputEngine):
    def generate(self, image_path: str, regions: List[Dict[str, Any]], output_dir: str, config: Dict[str, Any]):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Generate TXT
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        self._generate_txt(regions, txt_path)
        
        # 2. Generate PDF (if requested)
        if config.get('pdf', {}).get('output_format', 'pdf') in ['pdf', 'both']:
            pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
            self._generate_pdf(image_path, regions, pdf_path, config)
            
        # 3. Generate XML (if requested)
        # Always generate XML if not explicitly disabled, or if 'xml' or 'both' is in config
        # The user request said "Save PageXML for each file (or combined file) to output/filename.xml"
        # I'll assume we always want it or if configured.
        # Let's check config.
        # For now, I'll add it as a standard output.
        xml_path = os.path.join(output_dir, f"{base_name}.xml")
        self._generate_xml(image_path, regions, xml_path)
        
    def _generate_xml(self, image_path: str, regions: List[Dict[str, Any]], output_path: str):
        import datetime
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        # Basic PageXML 2019-07-15 structure
        ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
        ET.register_namespace("", ns)
        
        pcgts = ET.Element(f"{{{ns}}}PcGts")
        
        # Metadata
        metadata = ET.SubElement(pcgts, f"{{{ns}}}Metadata")
        ET.SubElement(metadata, f"{{{ns}}}Creator").text = "Subscript.py"
        ET.SubElement(metadata, f"{{{ns}}}Created").text = datetime.datetime.now().isoformat()
        ET.SubElement(metadata, f"{{{ns}}}LastChange").text = datetime.datetime.now().isoformat()
        
        # Page
        with Image.open(image_path) as im:
            w, h = im.size
            
        page = ET.SubElement(pcgts, f"{{{ns}}}Page", {
            "imageFilename": os.path.basename(image_path),
            "imageWidth": str(w),
            "imageHeight": str(h)
        })
        
        # TextRegions
        for i, region in enumerate(regions):
            text = region.get('text', '')
            bbox = region['bbox'] # x1, y1, x2, y2
            x1, y1, x2, y2 = bbox
            
            # Points string for Coords: "x1,y1 x2,y1 x2,y2 x1,y2"
            points = f"{x1},{y1} {x2},{y1} {x2},{y2} {x1},{y2}"
            
            text_region = ET.SubElement(page, f"{{{ns}}}TextRegion", {"id": f"r{i}"})
            ET.SubElement(text_region, f"{{{ns}}}Coords", {"points": points})
            
            text_equiv = ET.SubElement(text_region, f"{{{ns}}}TextEquiv")
            ET.SubElement(text_equiv, f"{{{ns}}}Unicode").text = text
            
        # Save
        xml_str = minidom.parseString(ET.tostring(pcgts)).toprettyxml(indent="    ")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        # logger.info(f"Saved PageXML to {output_path}")
        
    def _generate_txt(self, regions: List[Dict[str, Any]], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for region in regions:
                text = region.get('text', '')
                if text:
                    f.write(text + "\n")
                if text:
                    f.write(text + "\n")
        # logger.info(f"Saved TXT to {output_path}")

    def _generate_pdf(self, image_path: str, regions: List[Dict[str, Any]], output_path: str, config: Dict[str, Any]):
        try:
            c = canvas.Canvas(output_path)
            with Image.open(image_path) as im:
                w, h = im.size
                
            c.setPageSize((w, h))
            c.drawImage(image_path, 0, 0, width=w, height=h)
            
            # Invisible Text
            c.setFillColorRGB(0, 0, 0, 0) 
            
            for region in regions:
                text = region.get('text', '')
                if not text: continue
                
                bbox = region['bbox'] # (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox
                
                # ReportLab Y is bottom-up
                pdf_y = h - y2
                box_height = y2 - y1
                box_width = x2 - x1
                
                text_object = c.beginText()
                text_object.setTextRenderMode(3) # Invisible
                font_size = box_height * 0.8 # Approximate
                text_object.setFont("Helvetica", font_size)
                text_object.setTextOrigin(x1, pdf_y)
                
                # Stretch to fit width
                text_width = c.stringWidth(text, "Helvetica", font_size)
                if text_width > 0 and box_width > 0:
                    scale = (box_width / text_width) * 100
                    text_object.setHorizScale(scale)
                    
                text_object.textLine(text)
                c.drawText(text_object)
                
            c.save()
            # logger.info(f"Saved PDF to {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")

    def combine_pdfs(self, pdf_paths: List[str], output_path: str):
        """Combines multiple PDF files into a single PDF."""
        try:
            from pypdf import PdfWriter
            
            merger = PdfWriter()
            for pdf in pdf_paths:
                merger.append(pdf)
            
            merger.write(output_path)
            merger.close()
            logger.info(f"Saved Combined PDF to {output_path}")
        except Exception as e:
            logger.error(f"Failed to combine PDFs: {e}")
