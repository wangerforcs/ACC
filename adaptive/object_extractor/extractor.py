import spacy

# Load SpaCy model (auto download if needed)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Configuration
PLACEMENT_VERBS = {
    "put", "place", "move", "drop",
    "insert", "push", "slide"
}

GOAL_PREPS = {
    "on", "onto", "into", "in",
    "inside", "to", "toward"
}

CONTEXT_PREPS = {
    "between", "next", "under",
    "over", "behind", "beside"
}

STOP_GENERIC_NOUNS = {
    "object", "item", "thing"
}


def get_full_np(token):
    """
    Collect full noun phrase with left modifiers
    (handles multi-modifier cases like 'small red plastic cup')
    """
    words = []

    def collect_left(t):
        for child in t.lefts:
            if child.dep_ in ("amod", "compound", "det", "conj", "cc"):
                # Handle compound modifiers like "yellow and white"
                if child.dep_ == "conj" and child.pos_ == "ADJ":
                    # Add the conjunction and the conjunct
                    conj_words = []
                    for c in child.lefts:
                        if c.dep_ == "cc":  # coordinating conjunction (and, or)
                            conj_words.append(c.text)
                    conj_words.append(child.text)
                    words.extend(conj_words)
                elif child.dep_ == "cc":  # coordinating conjunction
                    words.append(child.text)
                else:
                    collect_left(child)
                    words.append(child.text)

    collect_left(token)
    words.append(token.text)
    return " ".join(words)


def clean(text):
    if text is None:
        return None

    text = text.lower()
    for article in ["the ", "a ", "an "]:
        if text.startswith(article):
            text = text[len(article):]

    return text.strip()


def normalize_object(obj):
    if obj is None:
        return None
    obj = clean(obj)
    if obj in STOP_GENERIC_NOUNS:
        return None
    return obj


def extract_contextual_phrase(token):
    """
    Extract contextual phrase for an object, including prepositional phrases
    (e.g., "black bowl between the plate and the ramekin")
    """
    # Start with the basic noun phrase
    base_phrase = get_full_np(token)
    
    # Check for prepositional phrases attached to this token
    for child in token.children:
        if child.dep_ == "prep":
            # Build the prepositional phrase
            prep_phrase = child.text
            
            # Collect all pobj and conj children
            pobj_children = []
            conj_children = []
            
            for prep_child in child.children:
                if prep_child.dep_ == "pobj":
                    pobj_children.append(prep_child)
                elif prep_child.dep_ == "conj" and prep_child.pos_ in ("NOUN", "PROPN"):
                    conj_children.append(prep_child)
            
            # Handle pobj children
            for pobj_child in pobj_children:
                # Extract full phrase for this pobj
                pobj_full = get_full_np(pobj_child)
                prep_phrase += " " + pobj_full
                
                # Check for conjunctions attached to this pobj
                for c in pobj_child.children:
                    if c.dep_ == "conj" and c.pos_ in ("NOUN", "PROPN"):
                        # Look for coordinating conjunction
                        cc_found = False
                        for cc in c.lefts:
                            if cc.dep_ == "cc":
                                prep_phrase += " " + cc.text
                                cc_found = True
                                break
                        if not cc_found:
                            prep_phrase += " and"
                        prep_phrase += " " + get_full_np(c)
                
                # Handle nested prepositional phrases
                for c in pobj_child.children:
                    if c.dep_ == "prep":
                        nested_prep = c.text
                        for nested_pobj in c.children:
                            if nested_pobj.dep_ == "pobj":
                                nested_prep += " " + get_full_np(nested_pobj)
                        if len(nested_prep) > len(c.text):
                            prep_phrase += " " + nested_prep
            
            # Combine base phrase with prepositional phrase
            if len(prep_phrase) > len(child.text):
                return base_phrase + " " + prep_phrase
    
    return base_phrase


def extract_objects(instruction):
    """
    Extract objects from instruction, including contextual descriptions
    Returns a list of extracted objects
    """
    doc = nlp(instruction)
    
    # Track extracted objects
    extracted = []
    
    # Helper function to add object if not already present
    def add_object(obj_phrase):
        normalized = normalize_object(obj_phrase)
        if normalized and normalized not in extracted:
            extracted.append(normalized)
    
    # First pass: extract main objects from verbs
    verbs = [t for t in doc if t.pos_ == "VERB"]
    
    for verb in verbs:
        # Find direct objects (main manipulated objects)
        dobjs = [child for child in verb.children if child.dep_ in ("dobj", "obj") and child.pos_ in ("NOUN", "PROPN")]
        
        for dobj in dobjs:
            # Extract full contextual phrase for the direct object
            full_phrase = extract_contextual_phrase(dobj)
            add_object(full_phrase)
        
        # Find prepositional objects (placement targets)
        for child in verb.children:
            if child.dep_ == "prep":
                pobjs = [c for c in child.children if c.dep_ == "pobj" and c.pos_ in ("NOUN", "PROPN")]
                
                for pobj in pobjs:
                    # Extract base phrase for placement targets (usually simpler)
                    base_phrase = get_full_np(pobj)
                    add_object(base_phrase)
    
    # Second pass: ensure we capture all contextual phrases
    # Look for noun phrases with prepositional attachments
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            # Check if this noun has prepositional attachments
            has_prep = any(child.dep_ == "prep" for child in token.children)
            
            if has_prep:
                # Extract full contextual phrase
                full_phrase = extract_contextual_phrase(token)
                add_object(full_phrase)
    
    # Third pass: handle special patterns like "both A and B"
    for token in doc:
        if token.text.lower() == "both":
            # Look for following noun phrases
            for child in token.children:
                if child.pos_ in ("NOUN", "PROPN"):
                    # Extract this noun phrase
                    add_object(get_full_np(child))
                    
                    # Look for "and" conjunction
                    for c in child.children:
                        if c.dep_ == "conj" and c.pos_ in ("NOUN", "PROPN"):
                            add_object(get_full_np(c))
    
    # Final filtering: keep only the most relevant phrases
    # Sort by length (longest first)
    extracted.sort(key=len, reverse=True)
    
    # Filter out phrases that are subsets of other phrases
    final_extracted = []
    for phrase in extracted:
        # Check if this phrase is not a subset of any already kept phrase
        is_redundant = False
        for kept in final_extracted:
            if phrase in kept:
                is_redundant = True
                break
        
        if not is_redundant:
            final_extracted.append(phrase)
    
    # Additional filtering to remove very generic phrases
    filtered = []
    for phrase in final_extracted:
        words = phrase.split()
        if len(words) > 1 or (len(words) == 1 and words[0] not in CONTEXT_PREPS and words[0] not in GOAL_PREPS):
            filtered.append(phrase)
    
    return filtered


def extract_objects_with_details(instruction):
    """
    Extract objects with detailed information
    Returns a dictionary with instruction and extracted objects
    """
    objects = extract_objects(instruction)
    return {
        "instruction": instruction,
        "extracted_objects": objects
    }