import difflib


def lcs_length(true_order, classified_order):
    """
    Calculate the length of the Longest Common Subsequence (LCS) using difflib.
    """
    matcher = difflib.SequenceMatcher(None, true_order, classified_order)
    lcs_len = sum(block.size for block in matcher.get_matching_blocks())
    return lcs_len


def count_inversions(sequence):
    """
    Efficiently count the number of inversions in a sequence using a Binary Indexed Tree (Fenwick Tree).
    This is equivalent to the Bubble Sort Distance when sorting in ascending order.
    """
    # Map each element to its index in the sorted order
    sorted_seq = {element: idx for idx, element in enumerate(sorted(sequence))}
    mapped_seq = [sorted_seq[element] for element in sequence]

    # Initialize BIT
    BIT = [0] * (len(sequence) + 1)

    def update(index):
        index += 1  # BIT is 1-indexed
        while index <= len(sequence):
            BIT[index] += 1
            index += index & -index

    def query(index):
        res = 0
        index += 1  # BIT is 1-indexed
        while index > 0:
            res += BIT[index]
            index -= index & -index
        return res

    inversions = 0
    for i in reversed(range(len(mapped_seq))):
        inversions += query(mapped_seq[i] - 1)
        update(mapped_seq[i])

    return inversions


def bubble_sort_distance(true_order, classified_order):
    """
    Calculate the number of adjacent swaps (Bubble Sort Distance) needed to transform
    the classified_order into the true_order.
    """
    # Create a mapping from element to its position in the true_order
    true_position = {element: idx for idx, element in enumerate(true_order)}
    # Transform classified_order to the order of true_order
    transformed = [true_position[element] for element in classified_order]
    # Count inversions in the transformed list
    return count_inversions(transformed)


def combined_normalized_loss(true_order, classified_order, alpha=0.5):
    """
    Combine normalized LCS loss and normalized Bubble Sort Distance into a single loss metric.
    """
    n = len(true_order)
    assert n == len(classified_order), "Both orders must have the same length."

    # Calculate LCS
    lcs_len = lcs_length(true_order, classified_order)
    lcs_loss = (n - lcs_len) / (n - 1)  # Normalize LCS loss to [0, 1]

    # Calculate BSD
    bsd = bubble_sort_distance(true_order, classified_order)
    max_bsd = n * (n - 1) / 2
    bsd_loss = bsd / max_bsd

    # Combine losses
    combined_loss = alpha * lcs_loss + (1 - alpha) * bsd_loss
    return combined_loss
