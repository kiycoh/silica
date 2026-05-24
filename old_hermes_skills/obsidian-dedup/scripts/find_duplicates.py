import os, json, argparse, sys, re

def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def are_similar(s1, s2):
    if s1 == s2:
        return True
    
    # Require minimum length for fuzzy matching
    if len(s1) < 4 or len(s2) < 4:
        return False
        
    # If one or both strings end with digits, they must have the exact same trailing digits
    m1 = re.search(r'\d+$', s1)
    m2 = re.search(r'\d+$', s2)
    if m1 or m2:
        d1 = m1.group() if m1 else ""
        d2 = m2.group() if m2 else ""
        if d1 != d2:
            return False
        
    # Quick length check: if length difference is > 1, they are not similar
    if abs(len(s1) - len(s2)) > 1:
        return False
        
    dist = edit_distance(s1, s2)
    if dist > 1:
        return False
        
    # Check for trailing Roman numerals difference (e.g. "fisica" vs "fisicai")
    if s1.startswith(s2) or s2.startswith(s1):
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        suffix = longer[len(shorter):]
        # If the suffix looks like a Roman numeral, reject
        roman_numerals = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'}
        if suffix in roman_numerals:
            return False
            
    return True

def has_leading_numbers(name):
    return bool(re.match(r'^\d+(?:\.\d+)*[\s\.\-\)\/]+', name))

def find_duplicates(vault_path, folder_path=None):
    search_path = folder_path if folder_path else vault_path
    
    # Map from normalized name to dict: {"paths": [...], "representative_name": "..."}
    groups = {}
    
    for root, dirs, files in os.walk(search_path):
        # Skip hidden folders like .git or .obsidian
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.md'):
                orig_name = os.path.splitext(file)[0]
                
                # Normalize name:
                # 1. Strip parenthetical suffixes (e.g., "Subword Tokenization (NLP)" -> "Subword Tokenization")
                norm = re.sub(r'\s*\(.*?\)\s*$', '', orig_name)
                # 2. Strip leading numbers/bullet points (e.g., "5. L'ambiente" -> "L'ambiente")
                # We ignore titles that are fully formatted as dates (e.g. YYYY-MM-DD or DD-MM-YYYY)
                # and limit digits to 1-3 to avoid stripping years.
                if not re.match(r'^\d{2,4}[-\.\/]\d{2}[-\.\/]\d{2,4}$', orig_name):
                    norm = re.sub(r'^\d{1,3}(?:\.\d{1,3})*[\s\.\-\)\/]+\s*', '', norm)
                # Fallback to orig_name if we ended up with an empty string
                if not norm:
                    norm = orig_name
                # 3. Lowercase
                norm = norm.lower()
                # 4. Strip non-alphanumeric characters
                norm = re.sub(r'[^a-z0-9]', '', norm)
                
                full_path = os.path.abspath(os.path.join(root, file))
                
                # Find an existing group key that is similar to `norm`
                matched_key = None
                for existing_key in groups:
                    if are_similar(norm, existing_key):
                        matched_key = existing_key
                        break
                
                if matched_key is None:
                    groups[norm] = {
                        "paths": [],
                        "representative_name": orig_name
                    }
                    matched_key = norm
                
                groups[matched_key]["paths"].append(full_path)
                
                # Keep the best representative name (prefer no leading numbers, then shorter name)
                current_rep = groups[matched_key]["representative_name"]
                pref_orig = not has_leading_numbers(orig_name)
                pref_curr = not has_leading_numbers(current_rep)
                if pref_orig != pref_curr:
                    if pref_orig:
                        groups[matched_key]["representative_name"] = orig_name
                else:
                    if len(orig_name) < len(current_rep):
                        groups[matched_key]["representative_name"] = orig_name
                        
    # Filter only duplicates and map to their representative names
    duplicates = {}
    for norm, group in groups.items():
        if len(group["paths"]) > 1:
            duplicates[group["representative_name"]] = group["paths"]
            
    return duplicates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault root")
    parser.add_argument("--folder", required=False, help="Subdirectory inside the vault to restrict the search")
    args = parser.parse_args()
    
    if not os.path.isdir(args.vault):
        print(json.dumps({"error": f"Vault path {args.vault} is not a directory"}))
        sys.exit(1)
        
    search_folder = None
    if args.folder:
        search_folder = os.path.abspath(args.folder)
        if not os.path.isdir(search_folder):
            print(json.dumps({"error": f"Folder path {args.folder} is not a directory"}))
            sys.exit(1)
        
    dupes = find_duplicates(args.vault, search_folder)
    print(json.dumps(dupes, indent=2))

