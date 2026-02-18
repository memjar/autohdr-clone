"""
IMI Batch Training Runner — Zero API cost, all local Ollama.
Generates training pairs from seeds + variations + learning log.
Run: python3 imi_batch_trainer.py
"""

import json
import os
import sys
import time
import logging
import signal

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("batch_trainer")


class TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError("Query timed out")

def run_with_timeout(func, timeout_sec=360):
    """Run a function with a hard timeout."""
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = func()
        signal.alarm(0)
        return result
    except TimeoutError:
        logger.warning(f"  Hard timeout after {timeout_sec}s")
        return None
    finally:
        signal.signal(signal.SIGALRM, old)

TRAINING_OUTPUT = os.path.expanduser("~/.axe/klaus_training/imi_finetune.jsonl")
LEARNING_LOG = os.path.expanduser("~/.axe/klaus_data/imi_learning_log.jsonl")


def count_existing():
    if not os.path.exists(TRAINING_OUTPUT):
        return 0
    with open(TRAINING_OUTPUT) as f:
        return sum(1 for _ in f)


def get_existing_queries():
    queries = set()
    if os.path.exists(TRAINING_OUTPUT):
        with open(TRAINING_OUTPUT) as f:
            for line in f:
                try:
                    queries.add(json.loads(line).get("instruction", ""))
                except:
                    pass
    return queries


def generate_variations(seeds):
    """Generate rephrasings of seed queries for training diversity."""
    variations = []
    prefixes = [
        "Tell me about", "What does the data show for", "Analyze",
        "Break down", "What insights exist on", "Show me"
    ]
    for query, qtype in seeds:
        # Extract core topic
        for prefix in prefixes:
            if not query.lower().startswith(prefix.lower()):
                variation = f"{prefix} {query.lower().lstrip('what is the ').lstrip('which ').lstrip('how ')}"
                variations.append((variation, qtype))
                break  # One variation per seed
    return variations


def mine_learning_log():
    """Extract high-quality queries from the learning log."""
    if not os.path.exists(LEARNING_LOG):
        return []
    queries = []
    with open(LEARNING_LOG) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("has_bold_numbers") and entry.get("response_length", 0) > 300:
                    qtype = entry.get("classification", {}).get("type", "lookup")
                    queries.append((entry["query"], qtype))
            except:
                pass
    return queries


def main():
    from imi_training_generator import generate_training_pair, SEED_QUERIES
    from imi_rag import query as rag_query

    existing = get_existing_queries()
    logger.info(f"Starting batch training. {len(existing)} existing pairs.")

    # Phase 1: All seeds
    all_queries = list(SEED_QUERIES)

    # Phase 2: Variations
    all_queries.extend(generate_variations(SEED_QUERIES[:20]))

    # Phase 3: Learning log
    all_queries.extend(mine_learning_log())

    # Deduplicate
    seen = set(existing)
    unique_queries = []
    for q, t in all_queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append((q, t))

    logger.info(f"{len(unique_queries)} new queries to process.")

    generated = 0
    failed = 0

    for i, (query, qtype) in enumerate(unique_queries):
        try:
            logger.info(f"[{i+1}/{len(unique_queries)}] {query[:60]}...")
            chunks = rag_query(query, n=15)
            if not chunks:
                logger.warning(f"  No RAG chunks, skipping.")
                failed += 1
                continue

            pair = run_with_timeout(
                lambda q=query, t=qtype, c=chunks: generate_training_pair(q, t, c),
                timeout_sec=360
            )
            if pair:
                os.makedirs(os.path.dirname(TRAINING_OUTPUT), exist_ok=True)
                with open(TRAINING_OUTPUT, "a") as f:
                    f.write(json.dumps(pair) + "\n")
                generated += 1
                logger.info(f"  OK (quality={pair['metadata']['quality_score']}, len={pair['metadata']['response_length']})")
            else:
                failed += 1
                logger.warning(f"  Quality gate failed or timed out.")
        except Exception as e:
            failed += 1
            logger.warning(f"  Error: {e}")

        # Brief pause to avoid overwhelming Ollama
        time.sleep(2)

    total = count_existing()
    logger.info(f"DONE. Generated={generated}, Failed={failed}, Total in file={total}")


if __name__ == "__main__":
    main()
