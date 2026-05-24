import argparse
import json
import os
import sys

# --- hermes_common bootstrap (uniform across all hermes skills) ---
_p = os.path.dirname(os.path.abspath(__file__))
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "hermes_common")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)
# --- end bootstrap ---

from hermes_common import templates

def write_note(path, content):
    # Use OS native solution for writing notes
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Failed to write note to {path}: {e}", file=sys.stderr)
        return False

def read_note(path):
    # Use OS native solution — hermes_tools.read_file returns LINE|CONTENT
    # format that corrupts patch ops when embedded verbatim.
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Failed to read note from {path}: {e}", file=sys.stderr)
    return None

def delete_note(path):
    """Delete a note verbatim from filesystem."""
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception as e:
        print(f"Failed to delete note {path}: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Bulk note writer and patcher for Hermes obsidian-injector")
    parser.add_argument("--operations", required=True, help="Path to JSON file containing operations")
    args = parser.parse_args()

    if not os.path.exists(args.operations):
        print(json.dumps({"error": f"Operations file {args.operations} does not exist"}))
        sys.exit(1)

    try:
        with open(args.operations, 'r', encoding='utf-8') as f:
            ops = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Failed to parse operations JSON: {e}"}))
        sys.exit(1)

    results = []
    success_count = 0

    for idx, op in enumerate(ops):
        op_type = op.get("op")
        path = op.get("path")
        
        if not path:
            results.append({"index": idx, "success": False, "error": "Missing 'path' parameter"})
            continue

        if op_type == "write":
            heading = op.get("heading")
            snippet = op.get("snippet", "")
            hub = op.get("hub")
            tags = op.get("tags")
            related = op.get("related")

            if not heading or not hub:
                results.append({"index": idx, "path": path, "success": False, "error": "Missing 'heading' or 'hub' parameter for write operation"})
                continue

            content = templates.template_spoke(
                heading=heading,
                snippet=snippet,
                hub=hub,
                tags=tags,
                related=related
            )
            
            ok = write_note(path, content)
            if ok:
                success_count += 1
            results.append({"index": idx, "path": path, "op": "write", "success": ok})

        elif op_type == "patch":
            heading = op.get("heading")
            snippet = op.get("snippet")
            source_basename = op.get("source_basename")
            hub = op.get("hub")

            if not heading or not snippet or not source_basename:
                results.append({"index": idx, "path": path, "success": False, "error": "Missing 'heading', 'snippet', or 'source_basename' for patch operation"})
                continue

            existing_content = read_note(path)
            if existing_content is None:
                results.append({"index": idx, "path": path, "success": False, "error": "Cannot patch; target file does not exist"})
                continue
            
            new_content = templates.patch_snippet(
                heading=heading,
                snippet=snippet,
                source_basename=source_basename,
                hub=hub,
                existing_content=existing_content
            )
            
            ok = write_note(path, new_content)
            if ok:
                success_count += 1
            results.append({"index": idx, "path": path, "op": "patch", "success": ok})

        elif op_type == "overwrite":
            content = op.get("content")
            if content is None:
                results.append({"index": idx, "path": path, "success": False,
                                "error": "Missing 'content' for overwrite operation"})
                continue
            ok = write_note(path, content)
            if ok:
                success_count += 1
            results.append({"index": idx, "path": path, "op": "overwrite", "success": ok})

        elif op_type == "delete":
            ok = delete_note(path)
            if ok:
                success_count += 1
            results.append({"index": idx, "path": path, "op": "delete", "success": ok})

        else:
            results.append({"index": idx, "path": path, "success": False, "error": f"Unknown operation type: {op_type}"})

    report = {
        "success": success_count == len(ops),
        "total_operations": len(ops),
        "successful_operations": success_count,
        "results": results
    }
    
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
