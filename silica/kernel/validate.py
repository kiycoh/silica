import os
from silica.driver import DRIVER

def validate_operations(ops: list, payloads: list, target_dir: str) -> tuple[list, list]:
    """Validates operations against payloads and target_dir using DRIVER."""
    
    valid_concepts = {}
    expected_collision_paths = {}
    inbox_folders = set()

    # Index payloads
    for payload_data in payloads:
        batches = payload_data.get("batches", [])
        for batch in batches:
            inbox_file = batch.get("inbox_file")
            if not inbox_file:
                continue
                
            source_basename = os.path.basename(inbox_file)
            inbox_dir = os.path.dirname(os.path.abspath(inbox_file))
            inbox_folders.add(inbox_dir)
            
            if source_basename not in valid_concepts:
                valid_concepts[source_basename] = set()
                
            for c in batch.get("concepts", []):
                name = c.get("name")
                if not name:
                    continue
                valid_concepts[source_basename].add(name)
                
                collision = c.get("vault_collision")
                if collision and isinstance(collision, dict) and collision.get("path"):
                    expected_collision_paths[(source_basename, name)] = collision["path"]
                else:
                    expected_collision_paths[(source_basename, name)] = None

    # Helper to check existence via DRIVER
    def path_exists(p: str) -> bool:
        try:
            DRIVER.read_note(p)
            return True
        except RuntimeError:
            return False

    # 1. Coerce write <-> patch
    for op in ops:
        op_type = op.get("op")
        path = op.get("path")
        source_basename = op.get("source_basename")
        heading = op.get("heading")
        if op_type == "write" and path and path_exists(path):
            op["op"] = "patch"
        elif op_type == "patch" and path and not path_exists(path):
            expected_path = expected_collision_paths.get((source_basename, heading))
            if not expected_path or os.path.abspath(path) == os.path.abspath(expected_path):
                op["op"] = "write"

    # 2. Global deduplication
    path_groups = {}
    for op in ops:
        op_type = op.get("op")
        path = op.get("path")
        if op_type in ("write", "patch") and path:
            norm_path = os.path.abspath(path)
            if norm_path not in path_groups:
                path_groups[norm_path] = []
            path_groups[norm_path].append(op)

    for norm_path, group in path_groups.items():
        if len(group) > 1:
            richest_op = max(group, key=lambda o: len(o.get("snippet", "")))
            for op in group:
                if op is not richest_op:
                    op["op"] = "skip"
                    op["reason"] = f"Duplicate write/patch to the same path '{op.get('path')}'"

    validated_ops = []
    rejected_ops = []
    
    target_dir_abs = os.path.abspath(target_dir) if target_dir else ""

    for idx, op in enumerate(ops):
        heading = op.get("heading")
        op_type = op.get("op")
        source_basename = op.get("source_basename")
        path = op.get("path")
        
        if not heading or not op_type:
            rejected_ops.append({"op": op, "reason": "Missing 'heading' or 'op' field"})
            continue

        if not source_basename:
            rejected_ops.append({"op": op, "reason": "Missing 'source_basename' field"})
            continue

        if source_basename not in valid_concepts:
            rejected_ops.append({"op": op, "reason": f"Unknown source_basename '{source_basename}'"})
            continue

        if heading not in valid_concepts[source_basename]:
            rejected_ops.append({"op": op, "reason": f"Heading '{heading}' not present in payload concepts"})
            continue

        if path:
            path_abs = os.path.abspath(path)
            forbidden = any(path_abs.startswith(folder) for folder in inbox_folders)
            if "/0 Inbox/" in path or "/0 inbox/" in path.lower() or forbidden:
                rejected_ops.append({"op": op, "reason": f"Target path '{path}' contains forbidden inbox segment"})
                continue

        if op_type == "skip":
            continue
            
        elif op_type == "patch":
            if not path:
                rejected_ops.append({"op": op, "reason": "Missing 'path' field for patch operation"})
                continue
                
            expected_path = expected_collision_paths.get((source_basename, heading))
            path_abs = os.path.abspath(path)
            
            if expected_path:
                if path_abs != os.path.abspath(expected_path):
                    rejected_ops.append({"op": op, "reason": f"Path '{path}' does not match expected collision '{expected_path}'"})
                    continue
            else:
                if target_dir_abs and not path_abs.startswith(target_dir_abs):
                    rejected_ops.append({"op": op, "reason": f"Coerced patch path '{path}' not in target folder"})
                    continue

            if not path_exists(path):
                rejected_ops.append({"op": op, "reason": f"Collision path '{path}' does not exist in vault"})
                continue

            validated_ops.append(op)

        elif op_type == "write":
            if not path:
                rejected_ops.append({"op": op, "reason": "Missing 'path' field for write operation"})
                continue
                
            path_abs = os.path.abspath(path)
            if target_dir_abs and not path_abs.startswith(target_dir_abs):
                rejected_ops.append({"op": op, "reason": f"Path '{path}' not in target folder"})
                continue

            if path_exists(path):
                rejected_ops.append({"op": op, "reason": f"Target path '{path}' already exists (should be patch)"})
                continue

            validated_ops.append(op)

        else:
            rejected_ops.append({"op": op, "reason": f"Unknown operation type '{op_type}'"})

    return validated_ops, rejected_ops
