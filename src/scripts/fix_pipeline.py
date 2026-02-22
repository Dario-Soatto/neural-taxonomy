import os

# --- FILE PATHS ---
STEP2_FILE = "src/step_2__create_supervised_similarity_data.py"
UTILS_FILE = "src/utils_vllm_client.py"

# --- 1. PATCH UTILS_VLLM_CLIENT.PY (Disable Structured Outputs to stop leaks) ---
# We will effectively "turn off" the feature by forcing the import to fail
# or by removing the logic block that uses it.
def patch_utils():
    if not os.path.exists(UTILS_FILE):
        print(f"Error: {UTILS_FILE} not found.")
        return

    with open(UTILS_FILE, 'r') as f:
        content = f.read()

    # Disable the import of StructuredOutputsParams to prevent usage
    new_content = content.replace(
        "from vllm.sampling_params import StructuredOutputsParams",
        "# from vllm.sampling_params import StructuredOutputsParams # DISABLED due to leaks"
    )
    
    # Also disable the usage block if it wasn't caught by import check
    if "if response_format is not None and StructuredOutputsParams is not None:" in new_content:
        print("Disabling StructuredOutputs logic block...")
        new_content = new_content.replace(
            "if response_format is not None and StructuredOutputsParams is not None:",
            "if False and response_format is not None and StructuredOutputsParams is not None: # DISABLED"
        )

    with open(UTILS_FILE, 'w') as f:
        f.write(new_content)
    print(f"✅ patched {UTILS_FILE} (Disabled Structured Outputs)")

# --- 2. PATCH STEP 2 (Add Nuclear Regex Parser) ---
def patch_step2():
    if not os.path.exists(STEP2_FILE):
        print(f"Error: {STEP2_FILE} not found.")
        return

    with open(STEP2_FILE, 'r') as f:
        content = f.read()

    # The robust regex parser
    new_parser = """def _parse_similarity_pairs(raw_text):
    if raw_text is None:
        return None
    
    # REGEX: Extract pairs ignoring JSON syntax completely
    # Matches keys/values with single OR double quotes
    pairs = []
    # Looks for pair_idx: <num> ... label: <Yes/No>
    pattern = r'pair_idx[\"\']?\\s*:\\s*(\\d+).*?label[\"\']?\\s*:\\s*[\"\']?([Yy]es|[Nn]o)[\"\']?'
    
    import re
    for match in re.finditer(pattern, raw_text, re.DOTALL | re.IGNORECASE):
        pairs.append({
            "pair_idx": int(match.group(1)),
            "label": match.group(2).capitalize()
        })
        
    return pairs if pairs else None
"""
    
    # Replace the existing function
    # We find the start of the function and assume it ends before the next def
    # or just replace the previous attempt we made.
    
    if "def _parse_similarity_pairs" in content:
        # Simple string replacement for the function signature works best if we can identify the block
        # But since the previous content varies, let's append the NEW parser at the end 
        # and change the calls to use it.
        
        # 1. Add the new function to the end of imports
        import_end_idx = content.find("import ")
        content = new_parser + "\n" + content
        
        # 2. Rename the old function usage
        # We replace the CALL sites to use `_parse_similarity_pairs` which is now defined at top? 
        # No, Python needs definition before use.
        
        # BETTER STRATEGY: Read file, remove old function if strictly defined, or just append 
        # new one with a distinct name and update calls.
        
        new_func_name = "_parse_similarity_pairs_regex_v3"
        new_parser_renamed = new_parser.replace("_parse_similarity_pairs", new_func_name)
        
        # Append to end of file (before main check)
        content = content.replace("if __name__ ==", new_parser_renamed + "\n\nif __name__ ==")
        
        # Update all calls
        content = content.replace("pairs = _parse_similarity_pairs(raw_text)", f"pairs = {new_func_name}(raw_text)")
        content = content.replace("pairs = _parse_similarity_pairs_v2(raw_text)", f"pairs = {new_func_name}(raw_text)")
        
        with open(STEP2_FILE, 'w') as f:
            f.write(content)
        print(f"✅ patched {STEP2_FILE} (Added Regex Parser)")
    else:
        print("Could not find function to patch in Step 2.")

if __name__ == "__main__":
    patch_utils()
    patch_step2()