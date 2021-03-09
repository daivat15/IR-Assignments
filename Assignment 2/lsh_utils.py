# Hash function for hashing shingles in a dictionary
def shingle_hash(shingle):
    hash = ""
    for slice in shingle:
        hash = hash + "_" + slice
    return hash