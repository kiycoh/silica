import os
import logging
from pydantic import BaseModel
from silica.driver import DRIVER
from silica.kernel.ops import Op, OpType

logger = logging.getLogger(__name__)


class Rejection(BaseModel):
    op: Op
    reason: str


def validate_operations(ops: list[Op] | list[dict], payloads: list, target_dir: str, hub: str | None = None) -> tuple[list[Op], list[Rejection]]:
    """Validates operations against payloads and target_dir using DRIVER."""
    from silica.kernel.ops_io import parse_ops
    ops = parse_ops(ops)
    valid_concepts: dict[str, set[str]] = {}
    expected_collision_paths: dict[tuple[str, str], str | None] = {}
    inbox_folders = set()
    has_payloads = bool(payloads)

    # Index payloads
    if has_payloads:
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

    # 1. Coerce write <-> patch and enforce default hub fallback
    if not hub and target_dir:
        hub = os.path.basename(target_dir.rstrip("/\\"))

    for op in ops:
        if op.op in (OpType.write, OpType.patch, OpType.overwrite) and hub:
            op.hub = hub
            
        if op.op == OpType.write and op.path and path_exists(op.path):
            op.op = OpType.patch
        elif op.op == OpType.patch and op.path and not path_exists(op.path):
            if has_payloads:
                expected_path = expected_collision_paths.get((op.source_basename, op.heading))
                if not expected_path or os.path.abspath(op.path) == os.path.abspath(expected_path):
                    op.op = OpType.write
            else:
                op.op = OpType.write

    # 2. Global deduplication
    path_groups: dict[str, list[Op]] = {}
    for op in ops:
        path = op.touched_ref()
        if path:
            norm_path = os.path.abspath(path)
            if norm_path not in path_groups:
                path_groups[norm_path] = []
            path_groups[norm_path].append(op)

    for norm_path, group in path_groups.items():
        if len(group) > 1:
            richest_op = max(group, key=lambda o: len(o.snippet or o.content or ""))
            has_write = any(o.op in (OpType.write, OpType.overwrite) for o in group)
            for op in group:
                if op is not richest_op:
                    op.op = OpType.skip
                    op.reason = f"Duplicate operation to the same path '{op.path}'"
            if has_write:
                # If there's an overwrite in the group, richest_op becomes overwrite
                if any(o.op == OpType.overwrite for o in group):
                    richest_op.op = OpType.overwrite
                else:
                    richest_op.op = OpType.write

    validated_ops = []
    rejected_ops = []
    
    target_dir_abs = os.path.abspath(target_dir) if target_dir else ""

    for op in ops:
        heading = op.heading
        op_type = op.op
        source_basename = op.source_basename
        path = op.path

        if has_payloads:
            if not heading:
                rejected_ops.append(Rejection(op=op, reason="Missing 'heading' field"))
                continue
            if not source_basename:
                rejected_ops.append(Rejection(op=op, reason="Missing 'source_basename' field"))
                continue
            if source_basename not in valid_concepts:
                rejected_ops.append(Rejection(op=op, reason=f"Unknown source_basename '{source_basename}'"))
                continue
            if heading not in valid_concepts[source_basename]:
                rejected_ops.append(Rejection(op=op, reason=f"Heading '{heading}' not present in payload concepts"))
                continue

        if path:
            path_abs = os.path.abspath(path)
            forbidden = any(path_abs.startswith(folder) for folder in inbox_folders)
            if "/0 Inbox/" in path or "/0 inbox/" in path.lower() or forbidden:
                rejected_ops.append(Rejection(op=op, reason=f"Target path '{path}' contains forbidden inbox segment"))
                continue

        if op_type == OpType.skip:
            continue
            
        elif op_type == OpType.patch:
            if not path:
                rejected_ops.append(Rejection(op=op, reason="Missing 'path' field for patch operation"))
                continue
                
            path_abs = os.path.abspath(path)
            if has_payloads:
                expected_path = expected_collision_paths.get((source_basename, heading))
                if expected_path:
                    if path_abs != os.path.abspath(expected_path):
                        rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' does not match expected collision '{expected_path}'"))
                        continue
                else:
                    if target_dir_abs and not path_abs.startswith(target_dir_abs):
                        rejected_ops.append(Rejection(op=op, reason=f"Coerced patch path '{path}' not in target folder"))
                        continue

            if not path_exists(path):
                rejected_ops.append(Rejection(op=op, reason=f"Collision path '{path}' does not exist in vault"))
                continue

            validated_ops.append(op)

        elif op_type == OpType.write:
            if not path:
                rejected_ops.append(Rejection(op=op, reason="Missing 'path' field for write operation"))
                continue
                
            path_abs = os.path.abspath(path)
            if target_dir_abs and not path_abs.startswith(target_dir_abs):
                rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' not in target folder"))
                continue

            if path_exists(path):
                rejected_ops.append(Rejection(op=op, reason=f"Target path '{path}' already exists (should be patch/overwrite)"))
                continue

            validated_ops.append(op)

        elif op_type == OpType.overwrite:
            if not path:
                rejected_ops.append(Rejection(op=op, reason="Missing 'path' field for overwrite operation"))
                continue
            
            path_abs = os.path.abspath(path)
            if target_dir_abs and not path_abs.startswith(target_dir_abs):
                rejected_ops.append(Rejection(op=op, reason=f"Path '{path}' not in target folder"))
                continue

            if not path_exists(path):
                # If target note doesn't exist, overwrite degrades to write gracefully
                op.op = OpType.write
            
            validated_ops.append(op)

        else:
            rejected_ops.append(Rejection(op=op, reason=f"Unknown operation type '{op_type}'"))

    # 3. Auto-create missing Hub notes
    hubs_to_check = set()
    for op in validated_ops:
        op_type = op.op
        if op_type in (OpType.write, OpType.patch, OpType.overwrite):
            hub = op.hub
            if hub:
                clean_hub = hub.strip("[]")
                if clean_hub:
                    hubs_to_check.add(clean_hub)

    hub_ops = []
    for hub in sorted(hubs_to_check):
        if not path_exists(hub):
            hub_filename = f"{hub}.md"
            hub_path = os.path.join(target_dir, hub_filename).replace("\\", "/")
            
            already_creating = any(
                (o.op == OpType.write and o.heading == hub) or
                (o.path and os.path.abspath(o.path) == os.path.abspath(hub_path))
                for o in validated_ops
            )
            
            if not already_creating:
                source_basename = "auto_generated"
                if validated_ops:
                    source_basename = validated_ops[0].source_basename or "auto_generated"
                
                hub_op = Op(
                    op=OpType.write,
                    heading=hub,
                    path=hub_path,
                    snippet=f"Hub automatically generated by the Injector pipeline.",
                    hub=hub,
                    source_basename=source_basename
                )
                hub_ops.append(hub_op)
                logger.info("Validation: hub '%s' does not exist. Injected creation operation at %s", hub, hub_path)

    if hub_ops:
        validated_ops = hub_ops + validated_ops

    return validated_ops, rejected_ops
