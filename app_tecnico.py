import os

path = "/Users/alfonsoariasaguilera/Desktop/trasteo/ibex35/app_tecnico.py"

if not os.path.exists(path):
    print(f"Error: File not found at {path}")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

if '"Stop Loss (€)": round(sl, 2),' in content:
    print("Column already exists. Skipping modification.")
else:
    lines = content.splitlines(keepends=True)
    new_lines = []
    modified = False
    for line in lines:
        new_lines.append(line)
        if '"Precio": round(close, 2),' in line and not modified:
            indent = line[:line.find('"')]
            new_lines.append(f'{indent}"Stop Loss (€)": round(sl, 2),\n')
            modified = True
            
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("Successfully added Stop Loss column.")
