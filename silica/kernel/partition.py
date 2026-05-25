import json

def partition_by_concepts(payload: dict, max_concepts: int, max_bytes: int = 80 * 1024) -> list:
    """Deterministic partition of payload into chunks.
    
    Each chunk is a payload dict of the form:
      {"schema_version": schema_version, "batches": [...]}
    such that:
      1. Total concept count in the chunk <= max_concepts (if max_concepts > 0)
      2. JSON-serialized size of the chunk <= max_bytes
    
    If a single concept itself exceeds max_bytes, it is placed in its own chunk.
    Order of batches and concepts is strictly preserved for determinism.
    """
    schema_version = payload.get("schema_version", 1)
    limit_concepts = max_concepts if max_concepts > 0 else 999999
    
    flat_concepts = []
    for batch in payload.get("batches", []):
        inbox_file = batch["inbox_file"]
        for concept in batch.get("concepts", []):
            flat_concepts.append((inbox_file, concept))
            
    if not flat_concepts:
        return []
        
    chunks = []
    current_concepts = []
    
    def build_chunk_dict(concept_list: list) -> dict:
        # Group list of (inbox_file, concept) into batches, preserving order
        batches_dict = {}
        for inbox_file, concept in concept_list:
            if inbox_file not in batches_dict:
                batches_dict[inbox_file] = []
            batches_dict[inbox_file].append(concept)
            
        batches = [
            {"inbox_file": k, "concepts": v}
            for k, v in batches_dict.items()
        ]
        return {"schema_version": schema_version, "batches": batches}
        
    for inbox_file, concept in flat_concepts:
        # Candidate chunk if we add this concept
        candidate_list = current_concepts + [(inbox_file, concept)]
        candidate_chunk = build_chunk_dict(candidate_list)
        
        # Check constraints
        candidate_size = len(json.dumps(candidate_chunk, ensure_ascii=False).encode('utf-8'))
        candidate_count = len(candidate_list)
        
        if candidate_count > limit_concepts or candidate_size > max_bytes:
            # If current_concepts is empty, it means even a single concept exceeds constraints.
            # We must output it as a single chunk to prevent infinite loop.
            if not current_concepts:
                chunks.append(candidate_chunk)
                current_concepts = []
            else:
                # Close current chunk, and start a new one with the current concept
                chunks.append(build_chunk_dict(current_concepts))
                
                # Check if the single concept itself exceeds constraints when in a new chunk
                single_chunk = build_chunk_dict([(inbox_file, concept)])
                single_size = len(json.dumps(single_chunk, ensure_ascii=False).encode('utf-8'))
                if single_size > max_bytes:
                    chunks.append(single_chunk)
                    current_concepts = []
                else:
                    current_concepts = [(inbox_file, concept)]
        else:
            current_concepts = candidate_list
            
    if current_concepts:
        chunks.append(build_chunk_dict(current_concepts))
        
    return chunks
