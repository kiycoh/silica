from silica.driver import DRIVER
from silica.kernel import templates

def execute_operations(ops: list) -> dict:
    results = []
    success_count = 0
    
    for idx, op in enumerate(ops):
        op_type = op.get("op")
        path = op.get("path")
        
        if not path:
            results.append({"index": idx, "success": False, "error": "Missing 'path' parameter"})
            continue
            
        try:
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
                
                # We use DRIVER.create
                DRIVER.create(path, content)
                success_count += 1
                results.append({"index": idx, "path": path, "op": "write", "success": True})
                
            elif op_type == "patch":
                heading = op.get("heading")
                snippet = op.get("snippet")
                source_basename = op.get("source_basename")
                hub = op.get("hub")

                if not heading or not snippet or not source_basename:
                    results.append({"index": idx, "path": path, "success": False, "error": "Missing 'heading', 'snippet', or 'source_basename' for patch operation"})
                    continue

                # Read existing content
                try:
                    nc = DRIVER.read_note(path)
                    existing_content = nc.content
                except RuntimeError as e:
                    results.append({"index": idx, "path": path, "success": False, "error": f"Cannot patch; {e}"})
                    continue

                new_content = templates.patch_snippet(
                    heading=heading,
                    snippet=snippet,
                    source_basename=source_basename,
                    hub=hub,
                    existing_content=existing_content
                )

                # Use overwrite() to preserve Obsidian's version history and block-refs.
                # delete+create is forbidden here: it destroys history (breaks rollback)
                # and severs block-references silently.
                DRIVER.overwrite(path, new_content)
                success_count += 1
                results.append({"index": idx, "path": path, "op": "patch", "success": True})
                
            elif op_type == "overwrite":
                content = op.get("content")
                if content is None:
                    results.append({"index": idx, "path": path, "success": False, "error": "Missing 'content' for overwrite operation"})
                    continue
                # Use overwrite() to preserve version history and block-refs
                DRIVER.overwrite(path, content)
                success_count += 1
                results.append({"index": idx, "path": path, "op": "overwrite", "success": True})
                
            elif op_type == "delete":
                DRIVER.delete(path)
                success_count += 1
                results.append({"index": idx, "path": path, "op": "delete", "success": True})
                
            else:
                results.append({"index": idx, "path": path, "success": False, "error": f"Unknown operation type: {op_type}"})
                
        except Exception as e:
            results.append({"index": idx, "path": path, "success": False, "error": str(e)})

    failed_count = len(ops) - success_count
    return {
        "success": failed_count == 0,
        "total_operations": len(ops),
        "successful_operations": success_count,
        "failed_operations": failed_count,  # explicit count for WRITE state check (B4)
        "results": results
    }
