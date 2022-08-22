---
Task: OCR
Tags:
  - OCR
  - Text extraction
  - Text recognition
---

# Ocr repo
The OCR model combines two models: 
- [PSNet](https://github.com/open-mmlab/mmocr/blob/main/configs/textdet/psenet/README.md): a text detection model that locates bounding boxes that contain texts and 
- [EasyOCR](https://github.com/JaidedAI/EasyOCR): a text recognition model that recognises texts in the detected bounding boxes.