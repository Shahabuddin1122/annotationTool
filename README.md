# 🎯 Python Application - Manual Annotation Tool

This project is a Python-based tool designed for manual point annotations (e.g., marking ball positions in images) and automatic Ball Detection Dataset Conversion:
From Bounding Box to Point-Based
Annotations. It supports environment setup, requirement installation, and easy tool launching.

---

## 📦 Setup Instructions

### 1. Create a Virtual Environment

Create a virtual environment in the project directory:
Activate the virtual environment:
```bash
python -m .venv .venv
```

```bash
venv\Scripts\activate for Windows

source venv/bin/activate for macOS/Linux
```
2. Install Requirements
Install all dependencies:

```bash
pip install -r requirements.txt
```
3. Run the Application
To start the automatic ball point detection, run:

```bash
python model.py
```

4. Manual Annotation Tool
To run the manual point annotation tool:

```bash
streamlit run manual_annotation.py
```


📁 Project Structure (Example)
```bash
Copy
Edit
annotationTool/
├── test_dataset                 # Input images and label
├── output/                      # Output YOLO labels
├── model.py                     # Main application entry point
├── manual_annotation.py         # Manual annotation interface
├── requirements.txt             # List of Python dependencies
└── README.md                    # This documentation
```

📬 Contact
Feel free to open an issue or reach out for any issues, suggestions, or collaboration!
