def chunkify(text, size=250, overlap=50):
    """
    Splits text into chunks of specified word size with overlap.
    """
    if overlap >= size:
        raise ValueError(f"O Overlap ({overlap}) nao pode ser maior ou igual ao Chunk Size ({size}). Escolha um Overlap menor.")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks
