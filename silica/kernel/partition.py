def partition_by_concepts(payload: dict, max_concepts: int) -> list:
    """Greedy bin-pack the payload's batches into chunks of <= max_concepts.
    
    If a single batch contains more than max_concepts, we split it into
    sub-batches with the same inbox_file but sliced concepts to keep
    each chunk <= max_concepts.
    """
    split_batches = []
    for batch in payload["batches"]:
        concepts = batch["concepts"]
        if len(concepts) <= max_concepts:
            split_batches.append(batch)
        else:
            for i in range(0, len(concepts), max_concepts):
                split_batches.append({
                    "inbox_file": batch["inbox_file"],
                    "concepts": concepts[i:i + max_concepts]
                })

    chunks = []
    current = []
    current_count = 0
    for batch in split_batches:
        batch_count = len(batch["concepts"])
        if current and (current_count + batch_count > max_concepts):
            chunks.append(current)
            current = []
            current_count = 0
        current.append(batch)
        current_count += batch_count
    if current:
        chunks.append(current)
    return [
        {"schema_version": payload.get("schema_version", 1), "batches": chunk}
        for chunk in chunks
    ]
