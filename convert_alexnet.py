import json
import os

notebooks = [
    "AlexNet_PyTorch.ipynb"
]

replacements = {
    "/content/drive/MyDrive/5flowersdata/flowers/train": "/home/nursena/Downloads/FLowers/flowers/train",
    "/content/drive/MyDrive/5flowersdata/flowers/val": "/home/nursena/Downloads/FLowers/flowers/val",
    "/content/drive/MyDrive/5flowersdata/flowers/": "/home/nursena/Downloads/FLowers/flowers/"
}

def convert_notebook(filename):
    print(f"Converting {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    output_filename = filename.replace(".ipynb", ".py")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, str):
                    source = source.splitlines(keepends=True)
                
                for line in source:
                    # Skip drive mounting lines
                    if "google.colab" in line or "drive.mount" in line:
                        f.write(f"# Skipped: {line}")
                        continue
                    
                    # Replace paths
                    for old, new in replacements.items():
                        line = line.replace(old, new)
                    
                    f.write(line)
                
                f.write("\n\n") # Separator between cells

    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    for nb in notebooks:
        if os.path.exists(nb):
            convert_notebook(nb)
        else:
            print(f"File not found: {nb}")
