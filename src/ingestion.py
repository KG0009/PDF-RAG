
import json
import re
from pathlib import Path

# File paths
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "bhumi_vikas_rules.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "chunks.jsonl"

# Chunk configuration
TARGET_CHARS = 1800
MAX_CHARS = 4500
PDF_PAGES = 108
VERSION = "2025-03-06"

# Identify numbered sub-rules and paragraph boundaries
SUBRULE = re.compile(
    r"(?m)^[ \t]{0,8}\(\d{1,3}\)[ \t]+(?=[A-Za-z\[])"
)

PARAGRAPH = re.compile(r"\n[ \t]*\n")

# Identify PDF running headers
PAGE_HEADER = re.compile(
    r"(?m)^[ \t]*Rule\s+[\dA-Za-z-]+"
    r"(?:\s*[–-]\s*\d+)?"
    r"[ \t]+M\.P\.\s+Bhumi\s+Vikas\s+Rules,"
    r"\s*2012[ \t]+\d+[ \t]*$"
)


def load_rules(path):
    """Load and validate the original JSONL rules."""

    required = {
        "rule", "title", "part",
        "page_start", "page_end", "text"
    }

    rules = []
    seen = set()

    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):

            if not line.strip():
                continue

            rule = json.loads(line)

            if not required.issubset(rule):
                raise ValueError(
                    f"Missing fields at line {line_no}"
                )

            if rule["rule"] in seen:
                raise ValueError(
                    f"Duplicate rule: {rule['rule']}"
                )

            if not isinstance(rule["text"], str) or not rule["text"].strip():
                raise ValueError(
                    f"Empty rule text at line {line_no}"
                )

            seen.add(rule["rule"])
            rules.append(rule)

    if not rules:
        raise ValueError("No rules found")

    return rules


def split_long_section(text, start, end):
    """Split oversized sections, preferring line boundaries."""

    sections = []

    while end - start > MAX_CHARS:

        limit = start + MAX_CHARS

        cut = text.rfind(
            "\n", start + TARGET_CHARS, limit
        )

        if cut == -1:
            cut = limit
        else:
            cut += 1

        sections.append((start, cut))
        start = cut

    if start < end:
        sections.append((start, end))

    return sections


def chunk_text(text):
    """Divide a rule into chunks while preserving its original text."""

    if len(text) <= MAX_CHARS:
        return [(0, len(text))]

    boundaries = {0, len(text)}

    # Prefer splitting at sub-rules
    boundaries.update(
        match.start()
        for match in SUBRULE.finditer(text)
    )

    # Also consider paragraph boundaries
    boundaries.update(
        match.end()
        for match in PARAGRAPH.finditer(text)
    )

    points = sorted(boundaries)
    units = []

    for start, end in zip(points, points[1:]):
        units.extend(
            split_long_section(text, start, end)
        )

    chunks = []
    start, end = units[0]

    for next_start, next_end in units[1:]:

        # Combine smaller sections when possible
        if (
            next_end - start <= MAX_CHARS
            and end - start < TARGET_CHARS
        ):
            end = next_end

        else:
            chunks.append((start, end))
            start, end = next_start, next_end

    chunks.append((start, end))

    # Avoid isolated PDF headers or tiny trailing chunks
    if len(chunks) > 1:

        last_start, last_end = chunks[-1]
        previous_start, _ = chunks[-2]

        last_text = text[last_start:last_end]

        if (
            (
                len(last_text) < 250
                or not PAGE_HEADER.sub("", last_text).strip()
            )
            and last_end - previous_start <= MAX_CHARS
        ):
            chunks[-2:] = [(previous_start, last_end)]

    return chunks


def ingest():
    """Convert original rule records into searchable RAG chunks."""

    rules = load_rules(INPUT_FILE)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output:

        for rule in rules:

            original_text = rule["text"]
            spans = chunk_text(original_text)

            # Ensure no original content is lost
            assert "".join(
                original_text[start:end]
                for start, end in spans
            ) == original_text

            start_page = int(rule["page_start"])

            # Correct extracted page 109 to actual final page 108
            end_page = min(
                int(rule["page_end"]),
                PDF_PAGES
            )

            if start_page < 1 or start_page > end_page:
                raise ValueError(
                    f"Invalid pages for Rule {rule['rule']}"
                )

            for index, (start, end) in enumerate(spans, start=1):

                exact_text = original_text[start:end]

                # Remove PDF running headers from searchable text only
                searchable_text = PAGE_HEADER.sub(
                    "", exact_text
                ).strip()

                chunk = {
                    "id": f"rule_{rule['rule']}_chunk_{index:03d}",
                    "rule": rule["rule"],
                    "title": rule["title"],
                    "part": rule["part"],
                    "page_start": start_page,
                    "page_end": end_page,
                    "page_range_type": "rule",
                    "source_page_end": rule["page_end"],
                    "version": VERSION,
                    "source": "data/bvn06032025E.pdf",
                    "char_start": start,
                    "char_end": end,
                    "text": exact_text,
                    "search_text": (
                        f"Rule {rule['rule']}: {rule['title']}\n"
                        f"{rule['part']}\n"
                        f"{searchable_text}"
                    )
                }

                output.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )

                total += 1

    print(f"Loaded {len(rules)} rules")
    print(f"Created {total} searchable chunks")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Source version: 6 March 2025")


if __name__ == "__main__":
    ingest()
