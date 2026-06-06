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
    deferred_stems: frozenset[str] = frozenset(),
) -> tuple[bool, list[str]]:
    """Verify that the changes do not introduce structural regressions.

    Rules (S3.2):
      1. Reject if unplanned orphans increase.
         An orphan is unplanned if it is in post.orphans, was NOT in pre.orphans,
         and was NOT explicitly created by this payload (created_paths).
      2. Reject if unresolved links from PRE-EXISTING notes increase.
         A new unresolved link is only a regression when its *source* is a
         pre-existing note. Ghost links from *newly created* notes are
         intentional forward references to concepts not yet in the vault —
         they mirror the same exemption that Rule 1 already grants to newly
         created orphans (unplanned_orphans = new_orphans - norm_created).
         Links to *deferred* targets (planned but not yet written due to a
         settle failure) are also exempt — they will be resolved on the next
         pipeline iteration when the deferred ops are retried.

    Args:
      deferred_stems: lowercase basenames (no extension) of notes that were
        planned in this chunk but failed to settle and were deferred for retry.

    Returns:
      (success, list_of_errors)
    """
    errors = []

    # 1. Unplanned orphans check
    norm_pre_orphans = {normalize_ref(ref) for ref in pre.orphans}
    norm_post_orphans = {normalize_ref(ref) for ref in post.orphans}
    norm_created = {normalize_path(p) for p in created_paths}

    # Notes we actually observed in the pre-snapshot neighborhood.
    # The incremental snapshot domain can grow between pre and post: new notes
    # bring their resolved link targets into the post-snapshot neighborhood even
    # though those targets were invisible at pre-snapshot time.  A pre-existing
    # orphan pulled in this way would appear as a false "new orphan" because it
    # was never in norm_pre_orphans.  We only flag regressions for notes we
    # have a concrete pre-write baseline for.
    norm_pre_observed = {normalize_path(k) for k in pre.link_counts}

    new_orphans = norm_post_orphans - norm_pre_orphans
    unplanned_orphans = (new_orphans & norm_pre_observed) - norm_created
    
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
    # Exempt links whose source is a newly created note — same carve-out that
    # Rule 1 grants to planned orphans. norm_created is already computed above.
    #
    # Also exempt sources not observed in the pre-snapshot domain: when a write
    # op creates a note that links to an existing vault note, that target's
    # neighborhood enters the post-snapshot but was absent from the pre-snapshot
    # (the newly created note didn't exist yet so its link targets weren't
    # included in the incremental domain).  Pre-existing ghost links on those
    # target notes surface as new_unres even though nothing changed in them —
    # a false positive that mirrors the Rule 1 domain-expansion problem already
    # guarded by norm_pre_observed.
    new_unres_blocking = {
        (src, tgt) for src, tgt in new_unres
        if src not in norm_created
        and src in norm_pre_observed   # must have a concrete pre-write baseline
        and tgt not in deferred_stems  # exempt planned-but-deferred targets
    }
    if new_unres_blocking:
        detail_links = []
        for link in post.unresolved:
            normalized = normalize_link(link.source, link.target)
            if normalized in new_unres_blocking:
                detail_links.append(f"[[{link.source.name}]] -> [[{link.target}]]")
        errors.append(f"New unresolved links introduced: {', '.join(detail_links)}")
        
    # 3. No broken pre-existing backlinks check
    pre_lower = {k.lower(): (k, v) for k, v in pre.backlink_counts.items()}
    post_lower = {k.lower(): v for k, v in post.backlink_counts.items()}
    shared_keys = set(pre_lower.keys()) & set(post_lower.keys())
    
    for norm_name in sorted(shared_keys):
        orig_name, pre_count = pre_lower[norm_name]
        post_count = post_lower[norm_name]
        if post_count < pre_count:
            errors.append(f"Broken backlinks detected for '{orig_name}': decreased from {pre_count} to {post_count}")
            
    success = len(errors) == 0
    return success, errors
