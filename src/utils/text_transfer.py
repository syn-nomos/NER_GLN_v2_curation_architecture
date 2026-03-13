import re

def get_safe_left_pull_index(text, annotations):
    """Finds a safe index to cut the end of 'text' to pull it."""
    matches = list(re.finditer(r'\S+\s*$', text))
    if not matches: return 0
    cut_idx = matches[-1].start()
    
    stable = False
    while not stable:
        stable = True
        for ann in annotations:
            if ann['start_offset'] < cut_idx < ann['end_offset']:
                cut_idx = ann['start_offset']
                stable = False
                
    while cut_idx > 0 and text[cut_idx-1].isspace():
        cut_idx -= 1
        
    return cut_idx

def get_safe_right_pull_index(text, annotations):
    """Finds a safe index to cut the start of 'text' to pull it."""
    match = re.search(r'^\s*\S+', text)
    if not match: return len(text)
    cut_idx = match.end()
    
    stable = False
    while not stable:
        stable = True
        for ann in annotations:
            if ann['start_offset'] < cut_idx < ann['end_offset']:
                cut_idx = ann['end_offset']
                stable = False
                
    while cut_idx < len(text) and text[cut_idx].isspace():
        cut_idx += 1
        
    return cut_idx

def transfer_from_left(prev_text, prev_anns, curr_text, curr_anns):
    if not prev_text.strip():
        return (prev_text, prev_anns), (curr_text, curr_anns)
        
    cut_idx = get_safe_left_pull_index(prev_text, prev_anns)
    if cut_idx == 0 and not prev_text[:cut_idx].strip():
        cut_idx = 0 # Pulling everything if it's just one word
        
    pulled_text = prev_text[cut_idx:]
    new_prev_text = prev_text[:cut_idx]
    
    new_prev_anns = []
    pulled_anns = []
    
    for ann in prev_anns:
        if ann['end_offset'] <= cut_idx:
            new_prev_anns.append(ann)
        elif ann['start_offset'] >= cut_idx:
            new_ann = dict(ann)
            new_ann['start_offset'] -= cut_idx
            new_ann['end_offset'] -= cut_idx
            pulled_anns.append(new_ann)
            
    join_str = " " if pulled_text and curr_text and not pulled_text.endswith(" ") and not curr_text.startswith(" ") else ""
    prefix = pulled_text + join_str
    
    new_curr_text = prefix + curr_text
    
    new_curr_anns = []
    for ann in pulled_anns:
        new_curr_anns.append(ann)
        
    for ann in curr_anns:
        new_ann = dict(ann)
        new_ann['start_offset'] += len(prefix)
        new_ann['end_offset'] += len(prefix)
        new_curr_anns.append(new_ann)
        
    return (new_prev_text, new_prev_anns), (new_curr_text, new_curr_anns)

def transfer_from_right(next_text, next_anns, curr_text, curr_anns):
    if not next_text.strip():
        return (next_text, next_anns), (curr_text, curr_anns)
        
    cut_idx = get_safe_right_pull_index(next_text, next_anns)
    
    pulled_text = next_text[:cut_idx]
    new_next_text = next_text[cut_idx:]
    
    new_next_anns = []
    pulled_anns = []
    
    for ann in next_anns:
        if ann['start_offset'] >= cut_idx:
            new_ann = dict(ann)
            new_ann['start_offset'] -= cut_idx
            new_ann['end_offset'] -= cut_idx
            new_next_anns.append(new_ann)
        elif ann['end_offset'] <= cut_idx:
            pulled_anns.append(dict(ann))
            
    join_str = " " if pulled_text and curr_text and not curr_text.endswith(" ") and not pulled_text.startswith(" ") else ""
    
    new_curr_text = curr_text + join_str + pulled_text
    shift_offset = len(curr_text) + len(join_str)
    
    new_curr_anns = []
    for ann in curr_anns:
        new_curr_anns.append(dict(ann))
        
    for ann in pulled_anns:
        new_ann = dict(ann)
        new_ann['start_offset'] += shift_offset
        new_ann['end_offset'] += shift_offset
        new_curr_anns.append(new_ann)
        
    return (new_next_text, new_next_anns), (new_curr_text, new_curr_anns)
