import sys
import json
import argparse
from pathlib import Path

def fix_links(files, hub):
    fixed = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            continue
        
        content = path.read_text(encoding='utf-8')
        link = f"[[{hub}]]"
        
        if link not in content:
            # Append a Relations section if not present, or just add the link
            if "# Relazioni" not in content:
                content += f"\\n\\n# Relazioni\\n- Hub: {link}"
            else:
                # Find the Relazioni section and add it if not there
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if line.strip() == "# Relazioni":
                        lines.insert(i + 1, f"- Hub: {link}")
                        break
                content = "\\n".join(lines)
            
            path.write_text(content, encoding='utf-8')
            fixed.append(file_path)
    
    return fixed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", required=True, help="JSON list of file paths")
    parser.add_argument("--hub", required=True, help="Hub name for the wikilink")
    args = parser.parse_args()
    
    files = json.loads(args.files)
    fixed_files = fix_links(files, args.hub)
    print(json.dumps({"fixed": fixed_files, "count": len(fixed_files)}))
