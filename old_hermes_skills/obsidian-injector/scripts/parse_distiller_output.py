import argparse
import json
import re
import sys
from pathlib import Path

def parse_json(raw: str, strict: bool):
    # Strip leading/trailing whitespaces and BOM
    cleaned = raw.strip()
    if cleaned.startswith('\ufeff'):
        cleaned = cleaned[1:]
    
    fence_pattern = re.compile(r'^```(?:json)?\s*\n(.*?)\n```$', re.DOTALL | re.IGNORECASE)
    inner_fence_pattern = re.compile(r'```(?:json)?\s*\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
    
    was_strict_clean = True
    processed = cleaned
    
    # Check if wrapped in fences
    m = fence_pattern.match(cleaned)
    if m:
        processed = m.group(1).strip()
        was_strict_clean = False
    else:
        # Check if fences are somewhere inside the string
        m = inner_fence_pattern.search(cleaned)
        if m:
            processed = m.group(1).strip()
            was_strict_clean = False
            
    # Parse the JSON
    parsed = None
    parse_err = None
    try:
        parsed = json.loads(processed)
    except json.JSONDecodeError as e:
        # Fallback: Find the first '{' or '[' and last '}' or ']'
        start_idx = -1
        for idx, ch in enumerate(raw):
            if ch in '{[':
                start_idx = idx
                break
        end_idx = -1
        for idx in range(len(raw) - 1, -1, -1):
            if raw[idx] in '}]':
                end_idx = idx
                break
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = raw[start_idx:end_idx+1]
            try:
                parsed = json.loads(candidate)
                was_strict_clean = False
            except json.JSONDecodeError as inner_e:
                parse_err = inner_e
        else:
            parse_err = e

    if parsed is None:
        raise parse_err

    if strict and not was_strict_clean:
        raise ValueError("Strict mode violation: markdown fences, preambles, or postambles were stripped from the output.")

    return parsed, was_strict_clean

def main():
    parser = argparse.ArgumentParser(description="Sanitize and parse Distiller JSON output")
    parser.add_argument("--in", required=True, dest="infile", type=Path, help="Input raw text file from Distiller")
    parser.add_argument("--out", type=Path, help="Output clean JSON file (defaults to stdout)")
    parser.add_argument("--strict", action="store_true", help="Exit with error code if any cleanup was performed")
    args = parser.parse_args()

    if not args.infile.exists():
        sys.stderr.write(f"Error: Input file {args.infile} does not exist\n")
        sys.exit(1)

    try:
        raw_content = args.infile.read_text(encoding="utf-8")
    except Exception as e:
        sys.stderr.write(f"Error reading file {args.infile}: {e}\n")
        sys.exit(1)

    try:
        parsed_obj, was_clean = parse_json(raw_content, args.strict)
    except Exception as e:
        sys.stderr.write(f"JSON Parse Error: {e}\n")
        sys.exit(1)

    rendered = json.dumps(parsed_obj, ensure_ascii=False, indent=2)
    if args.out:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"Error writing output to {args.out}: {e}\n")
            sys.exit(1)
    else:
        print(rendered)

if __name__ == "__main__":
    main()
