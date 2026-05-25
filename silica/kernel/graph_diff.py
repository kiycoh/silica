import logging
from silica.driver.base import GraphSnapshot, NoteRef

logger = logging.getLogger(__name__)

def normalize_ref(ref: NoteRef) -> str:
    path = ref.path or ref.name
    if path.endswith(".md"):
        path = path[:-3]
    path = path.replace("\\", "/")
    return path.strip("/").lower()

def normalize_path(p: str) -> str:
    if p.endswith(".md"):
        p = p[:-3]
    p = p.replace("\\", "/")
    return p.strip("/").lower()

def normalize_link(source_ref: NoteRef, target: str) -> tuple[str, str]:
    src = normalize_ref(source_ref)
    tgt = target.replace("\\", "/").strip("/").lower()
    if tgt.endswith(".md"):
        tgt = tgt[:-3]
    return (src, tgt)

def check_graph_regression(
    pre: GraphSnapshot,
    post: GraphSnapshot,
    created_paths: list[str],
) -> tuple[bool, list[str]]:
    """Verify that the changes do not introduce structural regressions.
    
    Rules (S3.2):
      1. Rifiuta se aumentano gli orfani non dichiarati (unplanned orphans).
         An orphan is unplanned if it is in post.orphans, was NOT in pre.orphans,
         and was NOT explicitly created by this payload (created_paths).
      2. Rifiuta se aumentano i link irrisolti (new unresolved links).
         Any unresolved link in post.unresolved that was not in pre.unresolved is rejected.
         
    Returns:
      (success, list_of_errors)
    """
    errors = []
    
    # 1. Unplanned orphans check
    norm_pre_orphans = {normalize_ref(ref) for ref in pre.orphans}
    norm_post_orphans = {normalize_ref(ref) for ref in post.orphans}
    norm_created = {normalize_path(p) for p in created_paths}
    
    new_orphans = norm_post_orphans - norm_pre_orphans
    unplanned_orphans = new_orphans - norm_created
    
    if unplanned_orphans:
        # Find the original NoteRefs for reporting
        detail_names = []
        for ref in post.orphans:
            if normalize_ref(ref) in unplanned_orphans:
                detail_names.append(ref.path or ref.name)
        errors.append(f"Unplanned orphans introduced: {', '.join(detail_names)}")
        
    # 2. New unresolved links check
    pre_unres = {normalize_link(link.source, link.target) for link in pre.unresolved}
    post_unres = {normalize_link(link.source, link.target) for link in post.unresolved}
    
    new_unres = post_unres - pre_unres
    if new_unres:
        detail_links = []
        for link in post.unresolved:
            normalized = normalize_link(link.source, link.target)
            if normalized in new_unres:
                detail_links.append(f"[[{link.source.name}]] -> [[{link.target}]]")
        errors.append(f"New unresolved links introduced: {', '.join(detail_links)}")
        
    # 3. No broken pre-existing backlinks check
    for name, pre_count in pre.backlink_counts.items():
        norm_name = name.lower()
        post_count = 0
        for p_name, p_count in post.backlink_counts.items():
            if p_name.lower() == norm_name:
                post_count = p_count
                break
        if post_count < pre_count:
            errors.append(f"Broken backlinks detected for '{name}': decreased from {pre_count} to {post_count}")
            
    success = len(errors) == 0
    return success, errors
