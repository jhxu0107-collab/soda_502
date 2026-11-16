"""
This part build helpers for MUC evaluation pipeline:
data loading, generation, parsing, scoring.

Data: MUC-4 event extraction in chat format, from the Hugging Face hub
(`fnl-es/muc4-chat`). Each example is a conversation:
  system    -> the extraction instruction (the same for every example)
  user      -> a news report
  assistant -> the human annotation, as a JSON array of events
"""

# we first import packages
import json
import re
# need installation if you don't have
from datasets import load_dataset

dataset = "fnl-es/muc4-chat"
roles = ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]

# Load data
# filter the data for speed
def load_muc4(n_train, n_test, max_input_tokens=600, seed=0):
    """Load MUC-4, keep the shorter documents, and sample train / test examples.
    Each example: {"docid": ..., "messages": [system, user, assistant]}
    """
    # get ds
    ds = load_dataset(dataset)
    # sample ds
    def sample(split, n):
        part = ds[split].filter(lambda ex: ex["n_input_tokens"] <= max_input_tokens)
        # shuffle
        part = part.shuffle(seed=seed).select(range(n))
        return [{"docid": ex["docid"], "messages": ex["messages"]} for ex in part]

    return sample("train", n_train), sample("test", n_test)


# the system and messages
def prompt_of(example):
    """The system and user messages: what the model is given."""
    return example["messages"][:2]

# GT
def gold_of(example):
    """The human annotation: the text of the assistant message."""
    return example["messages"][2]["content"]


# generation
def generate(pipe, examples, batch_size=4, max_new_tokens=256):
    """Generate outputs for a list of examples; returns the model's raw text."""
    # padding to get the same length
    pipe.tokenizer.padding_side = "left"  # batched generation needs left padding
    prompts = [prompt_of(ex) for ex in examples]

                             # max output token          # don't do sample = set temperature as 0 for reproducibility
    outputs = pipe(prompts, max_new_tokens=max_new_tokens, do_sample=False, batch_size=batch_size)
    return [out[0]["generated_text"][-1]["content"] for out in outputs]


# Evaluation: parsing and scoring
def parse_output(text):
    """Extract the JSON array of events from the model output;
     None if there is none."""
    # find the JSON part of the answer.
    # The model may add extra words before or after it, so we keep only the
    # text from the first "[" to the last "]". DOTALL lets it span several lines.
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        return None  # no brackets at all: not a usable answer

    # turn that text into Python data (a list of dicts).
    # If it is not valid JSON (missing bracket, single quotes, ...),
    # json.loads raises an error; we catch it
    try:
        events = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    # check the shape. Valid JSON is not enough: we need a list
    # in which every item is a dict (one dict per event). [] passes: no events.
    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
        return None

    return events  # a list of events, ready to compare with the annotation

# extract value & normalization for event type
def _event_types(events):
    return sorted(str(e.get("incident_type", "")).strip().lower() for e in events)


def _entities(events, role):
    """All entities filling a role in a document; each entity is a set of mention strings."""
    found = []
    for event in events:
        fillers = event.get(role, [])
        if not isinstance(fillers, list): # if not a list, pass
            continue
        for entity in fillers:
            mentions = entity if isinstance(entity, list) else [entity]
            mentions = {str(m).strip().lower() for m in mentions if str(m).strip()}
            if mentions and mentions not in found:
                found.append(mentions)
    return found


def _match(pred_entities, gold_entities):
    """Count predicted entities that share a mention with a not-yet-matched gold entity."""
    remaining = list(gold_entities)
    hits = 0
    for pred in pred_entities:
        for gold in remaining:
            if pred & gold:
                remaining.remove(gold)
                hits += 1
                break
    return hits


def score(raw_outputs, examples):
    """Compare model outputs with the human annotation.

    valid_json   share of outputs that parse as a JSON array of events
    event_types  share of documents where the list of incident types is exactly right
    role_f1      F1 over role fillers (who, what, with which weapon), pooled over documents
    exact_match  share of documents where the whole answer is identical to the annotation
    """
    n = len(examples)
    valid = types_ok = exact = 0
    hits = n_pred = n_gold = 0
    for text, example in zip(raw_outputs, examples):
        gold = json.loads(gold_of(example))
        pred = parse_output(text)
        for role in roles:
            n_gold += len(_entities(gold, role))
        if pred is None:
            continue
        valid += 1
        types_ok += _event_types(pred) == _event_types(gold)
        exact += pred == gold
        for role in roles:
            pred_entities, gold_entities = _entities(pred, role), _entities(gold, role)
            n_pred += len(pred_entities)
            hits += _match(pred_entities, gold_entities)
    precision = hits / n_pred if n_pred else 0.0
    recall = hits / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "valid_json": valid / n,
        "event_types": types_ok / n,
        "role_f1": f1,
        "exact_match": exact / n,
    }
