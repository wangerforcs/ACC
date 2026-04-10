"""
Mock Object Extractor for testing purposes.

`extract_objects()` keeps the legacy flat prompt list for DINO.
`extract_object_roles()` exposes task-role groupings for heuristics.
"""

# Predefined results for each instruction in libero.txt
PREDEFINED_RESULTS = {
    "open the middle drawer of the cabinet": ["middle drawer of the cabinet"],
    "open the top drawer and put the bowl inside": ["top drawer", "bowl"],
    "push the plate to the front of the stove": ["front of the stove", "plate"],
    "put the bowl on the plate": ["plate", "bowl"],
    "put the bowl on the stove": ["stove", "bowl"],
    "put the bowl on top of the cabinet": ["top of the cabinet", "bowl"],
    "put the cream cheese in the bowl": ["cream cheese", "bowl"],
    "put the wine bottle on the rack": ["wine bottle", "rack"],
    "put the wine bottle on top of the cabinet": ["top of the cabinet", "wine bottle"],
    "turn on the stove": ["stove"],
    "pick up the alphabet soup and place it in the basket": ["alphabet soup", "basket"],
    "pick up the bbq sauce and place it in the basket": ["bbq sauce", "basket"],
    "pick up the butter and place it in the basket": ["butter", "basket"],
    "pick up the chocolate pudding and place it in the basket": ["chocolate pudding", "basket"],
    "pick up the cream cheese and place it in the basket": ["cream cheese", "basket"],
    "pick up the ketchup and place it in the basket": ["ketchup", "basket"],
    "pick up the milk and place it in the basket": ["milk", "basket"],
    "pick up the orange juice and place it in the basket": ["orange juice", "basket"],
    "pick up the bbq sauce and place it in the moving basket": ["bbq sauce", "basket"],
    "pick up the bbq sauce and place it on the plate": ["bbq sauce", "plate"], #我新增的
    "pick up the alphabet soup and place it on the plate": ["alphabet soup", "plate"],
    "pick up the cream cheese and place it on the plate": ["cream cheese", "plate"],
    "pick up the salad dressing and place it in the basket": ["salad dressing", "basket"],
    "pick up the tomato sauce and place it in the basket": ["tomato sauce", "basket"],
    "pick up the black bowl between the plate and the ramekin and place it on the plate": ["black bowl between the plate and the ramekin", "plate"],
    "pick up the black bowl from table center and place it on the plate": ["black bowl from table center", "plate"],
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": ["black bowl in the top drawer of the wooden cabinet", "plate"],
    "pick up the black bowl next to the cookie box and place it on the plate": ["black bowl next to the cookie box", "plate"],
    "pick up the black bowl next to the plate and place it on the plate": ["black bowl next to the plate", "plate"],
    "pick up the black bowl next to the ramekin and place it on the plate": ["black bowl next to the ramekin", "plate"],
    "pick up the black bowl on the cookie box and place it on the plate": ["black bowl on the cookie box", "plate"],
    "pick up the black bowl on the ramekin and place it on the plate": ["black bowl on the ramekin", "plate"],
    "pick up the black bowl on the stove and place it on the plate": ["black bowl on the stove", "plate"],
    "pick up the black bowl on the wooden cabinet and place it on the plate": ["black bowl on the wooden cabinet", "plate"],
    "turn on the stove and put the moka pot on it": ["stove", "moka pot"],
    "put the black bowl in the bottom drawer of the cabinet and close it": ["bottom drawer of the cabinet", "black bowl"],
    "put the yellow and white mug in the microwave and close it": ["yellow and white mug", "microwave"],
    "put both moka pots on the stove": ["both moka pots", "stove"],
    "put both the alphabet soup and the cream cheese box in the basket": ["alphabet soup", "cream cheese box", "basket"],
    "put both the alphabet soup and the tomato sauce in the basket": ["alphabet soup", "tomato sauce", "basket"],
    "put both the cream cheese box and the butter in the basket": ["cream cheese box", "butter", "basket"],
    "put the white mug on the left plate and put the yellow and white mug on the right plate": ["white mug", "left plate", "yellow and white mug", "right plate"],
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": ["white mug", "plate", "chocolate pudding", "right of the plate"],
    "pick up the book and place it in the back compartment of the caddy": ["book", "back compartment of the caddy"]
}

PREDEFINED_ROLE_RESULTS = {
    "open the middle drawer of the cabinet": {"tograsp": [], "reference": ["middle drawer of the cabinet"]},
    "open the top drawer and put the bowl inside": {"tograsp": ["bowl"], "reference": ["top drawer"]},
    "push the plate to the front of the stove": {"tograsp": ["plate"], "reference": ["front of the stove"]},
    "put the bowl on the plate": {"tograsp": ["bowl"], "reference": ["plate"]},
    "put the bowl on the stove": {"tograsp": ["bowl"], "reference": ["stove"]},
    "put the bowl on top of the cabinet": {"tograsp": ["bowl"], "reference": ["top of the cabinet"]},
    "put the cream cheese in the bowl": {"tograsp": ["cream cheese"], "reference": ["bowl"]},
    "put the wine bottle on the rack": {"tograsp": ["wine bottle"], "reference": ["rack"]},
    "put the wine bottle on top of the cabinet": {"tograsp": ["wine bottle"], "reference": ["top of the cabinet"]},
    "turn on the stove": {"tograsp": [], "reference": ["stove"]},
    "pick up the alphabet soup and place it in the basket": {"tograsp": ["alphabet soup"], "reference": ["basket"]},
    "pick up the bbq sauce and place it in the basket": {"tograsp": ["bbq sauce"], "reference": ["basket"]},
    "pick up the butter and place it in the basket": {"tograsp": ["butter"], "reference": ["basket"]},
    "pick up the chocolate pudding and place it in the basket": {"tograsp": ["chocolate pudding"], "reference": ["basket"]},
    "pick up the cream cheese and place it in the basket": {"tograsp": ["cream cheese"], "reference": ["basket"]},
    "pick up the ketchup and place it in the basket": {"tograsp": ["ketchup"], "reference": ["basket"]},
    "pick up the milk and place it in the basket": {"tograsp": ["milk"], "reference": ["basket"]},
    "pick up the orange juice and place it in the basket": {"tograsp": ["orange juice"], "reference": ["basket"]},
    "pick up the bbq sauce and place it in the moving basket": {"tograsp": ["bbq sauce"], "reference": ["basket"]},
    "pick up the bbq sauce and place it on the plate": {"tograsp": ["bbq sauce"], "reference": ["plate"]},
    "pick up the alphabet soup and place it on the plate": {"tograsp": ["alphabet soup"], "reference": ["plate"]},
    "pick up the cream cheese and place it on the plate": {"tograsp": ["cream cheese"], "reference": ["plate"]},
    "pick up the salad dressing and place it in the basket": {"tograsp": ["salad dressing"], "reference": ["basket"]},
    "pick up the tomato sauce and place it in the basket": {"tograsp": ["tomato sauce"], "reference": ["basket"]},
    "pick up the black bowl between the plate and the ramekin and place it on the plate": {
        "tograsp": ["black bowl between the plate and the ramekin"],
        "reference": ["plate"],
    },
    "pick up the black bowl from table center and place it on the plate": {
        "tograsp": ["black bowl from table center"],
        "reference": ["plate"],
    },
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": {
        "tograsp": ["black bowl in the top drawer of the wooden cabinet"],
        "reference": ["plate"],
    },
    "pick up the black bowl next to the cookie box and place it on the plate": {
        "tograsp": ["black bowl next to the cookie box"],
        "reference": ["plate"],
    },
    "pick up the black bowl next to the plate and place it on the plate": {
        "tograsp": ["black bowl next to the plate"],
        "reference": ["plate"],
    },
    "pick up the black bowl next to the ramekin and place it on the plate": {
        "tograsp": ["black bowl next to the ramekin"],
        "reference": ["plate"],
    },
    "pick up the black bowl on the cookie box and place it on the plate": {
        "tograsp": ["black bowl on the cookie box"],
        "reference": ["plate"],
    },
    "pick up the black bowl on the ramekin and place it on the plate": {
        "tograsp": ["black bowl on the ramekin"],
        "reference": ["plate"],
    },
    "pick up the black bowl on the stove and place it on the plate": {
        "tograsp": ["black bowl on the stove"],
        "reference": ["plate"],
    },
    "pick up the black bowl on the wooden cabinet and place it on the plate": {
        "tograsp": ["black bowl on the wooden cabinet"],
        "reference": ["plate"],
    },
    "turn on the stove and put the moka pot on it": {"tograsp": ["moka pot"], "reference": ["stove"]},
    "put the black bowl in the bottom drawer of the cabinet and close it": {
        "tograsp": ["black bowl"],
        "reference": ["bottom drawer of the cabinet"],
    },
    "put the yellow and white mug in the microwave and close it": {
        "tograsp": ["yellow and white mug"],
        "reference": ["microwave"],
    },
    "put both moka pots on the stove": {"tograsp": ["both moka pots"], "reference": ["stove"]},
    "put both the alphabet soup and the cream cheese box in the basket": {
        "tograsp": ["alphabet soup", "cream cheese box"],
        "reference": ["basket"],
    },
    "put both the alphabet soup and the tomato sauce in the basket": {
        "tograsp": ["alphabet soup", "tomato sauce"],
        "reference": ["basket"],
    },
    "put both the cream cheese box and the butter in the basket": {
        "tograsp": ["cream cheese box", "butter"],
        "reference": ["basket"],
    },
    "put the white mug on the left plate and put the yellow and white mug on the right plate": {
        "tograsp": ["white mug", "yellow and white mug"],
        "reference": ["left plate", "right plate"],
    },
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": {
        "tograsp": ["white mug", "chocolate pudding"],
        "reference": ["plate", "right of the plate"],
    },
    "pick up the book and place it in the back compartment of the caddy": {
        "tograsp": ["book"],
        "reference": ["back compartment of the caddy"],
    },
}


def extract_objects(instruction):
    """
    Extract objects from instruction
    Returns predefined results for libero.txt instructions
    """
    # Normalize instruction to match keys
    normalized_instruction = instruction.strip().lower()
    
    # Return predefined result if found
    if normalized_instruction in PREDEFINED_RESULTS:
        return PREDEFINED_RESULTS[normalized_instruction]
    
    # Return empty list for unknown instructions
    return []


def _dedupe_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_object_roles(instruction):
    """
    Extract heuristic-oriented object roles from predefined mock mappings.
    """
    normalized_instruction = instruction.strip().lower()
    role_info = PREDEFINED_ROLE_RESULTS.get(normalized_instruction, None)
    if role_info is None:
        all_objects = extract_objects(instruction)
        tograsp = list(all_objects)
        reference = []
    else:
        tograsp = list(role_info.get("tograsp", []))
        reference = list(role_info.get("reference", []))
        all_objects = _dedupe_preserve_order(tograsp + reference)
        if not all_objects:
            all_objects = extract_objects(instruction)

    tograsp = _dedupe_preserve_order(tograsp)
    reference = _dedupe_preserve_order(reference)
    all_objects = _dedupe_preserve_order(tograsp + reference) or _dedupe_preserve_order(extract_objects(instruction))
    return {
        "instruction": instruction,
        "tograsp": tograsp,
        "reference": reference,
        "all_objects": all_objects,
    }


def extract_objects_with_details(instruction):
    """
    Extract objects with detailed information
    Returns a dictionary with instruction and extracted objects
    """
    role_info = extract_object_roles(instruction)
    objects = role_info["all_objects"]
    return {
        "instruction": instruction,
        "extracted_objects": objects,
        "tograsp_objects": role_info["tograsp"],
        "reference_objects": role_info["reference"],
    }


def get_full_np(token):
    """
    Mock function for compatibility
    """
    return str(token)


def clean(text):
    """
    Mock function for compatibility
    """
    return text


def normalize_object(obj):
    """
    Mock function for compatibility
    """
    return obj


def extract_contextual_phrase(token):
    """
    Mock function for compatibility
    """
    return str(token)
