DEFAULT_SEPARATORS = " -_.:/\\\t\r\n"


def normalize_plate_text(
    text: str,
    separators: str = DEFAULT_SEPARATORS,
) -> str:
    """Uppercase and remove benign separators without character correction."""

    return text.upper().translate(str.maketrans("", "", separators))


def levenshtein_distance(first: str, second: str) -> int:
    """Return the Levenshtein edit distance using bounded working memory."""

    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def normalized_edit_distance(first: str, second: str) -> float:
    denominator = max(len(first), len(second))
    if denominator == 0:
        return 0.0
    return levenshtein_distance(first, second) / denominator
