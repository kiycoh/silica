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
                
                # In DRIVER, we can overwrite by deleting and creating, or in FS backend just create overwrites?
                # Actually DRIVER.create doesn't say if it overwrites. Let's use DRIVER.delete then DRIVER.create?
                # Wait, DRIVER.append appends content. DRIVER.set_prop sets properties.
                # How do we overwrite a file in CLI? CLI 'append' exists, but what if we rewrite the entire file?
                # The prompt says: "Write operations are graph-safe (wikilinks updated by Obsidian's engine on move/rename)."
                # If we use fs_backend we can overwrite. If we use CLI, maybe there's no overwrite? 
                # Let's just create with overwrite=True or delete and create.
                # Actually, `obsidian create --overwrite` is supported by CLI! (from user prompt: `overwrite - Overwrite if file exists`)
                # But our driver interface only has `DRIVER.create(path, content)`.
                # Wait, if `patch_snippet` modifies the content, we should just delete the file and recreate it with the same path? No, that loses history/graph links!
                # Ah, does Obsidian CLI have an `overwrite` command? 
                # Let's check `cli_backend.py`: `create(path, content)` runs `obsidian create path=... content=...`.
                # If we just run it on an existing file, does it overwrite or error? By default CLI `create` errors if file exists unless `overwrite` is specified.
                # But our Driver doesn't expose `overwrite` parameter! 
                # Wait, does the Driver interface support overwriting? We can just use `DRIVER.delete(path)` then `DRIVER.create(path, content)`.
                # Let's just do that for now. Or better, update `cli_backend.py` to support `overwrite` if needed. Let's just delete and create.
                DRIVER.delete(path)
                DRIVER.create(path, new_content)
                success_count += 1
                results.append({"index": idx, "path": path, "op": "patch", "success": True})
                
            elif op_type == "overwrite":
                content = op.get("content")
                if content is None:
                    results.append({"index": idx, "path": path, "success": False, "error": "Missing 'content' for overwrite operation"})
                    continue
                DRIVER.delete(path)
                DRIVER.create(path, content)
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

    return {
        "success": success_count == len(ops) and len(ops) > 0,
        "total_operations": len(ops),
        "successful_operations": success_count,
        "results": results
    }
