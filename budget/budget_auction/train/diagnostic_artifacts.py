"""Content-bound diagnostic inputs; no numerical work or final-test selection."""
import hashlib
import json
import pickle
from pathlib import Path


TEACHER_FILES = ("identity.json", "frozen_selection.json", "frozen_selection.pkl")
MATCHED_CONTROLS = ("training", "trials", "policy_hidden", "audit_hidden_sizes",
    "screen_batches", "screen_samples", "screen_ic_threshold_sum", "shortlist_mode",
    "audit_candidates", "audit_candidates_per_trial", "audit", "final_batches",
    "final_samples", "ic_threshold_sum")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def config_digest(config):
    return sha256_bytes(json.dumps(config, sort_keys=True).encode())


def read_verified(path, expected):
    data = Path(path).read_bytes()
    if sha256_bytes(data) != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return data


def load_teacher(directory, pins, student, source_sha256):
    """Read exactly the three frozen validation artifacts, never final_result.

    Pins must be recorded in the diagnostic configuration before execution.
    A source mismatch requires a separately reviewed migration, not a silent load.
    """
    if set(pins) != set(TEACHER_FILES):
        raise ValueError("Pin all three teacher validation artifacts")
    raw = {name: read_verified(Path(directory)/name, pins[name]) for name in TEACHER_FILES}
    identity = json.loads(raw["identity.json"])
    teacher = identity["config"]
    if identity["config_sha256"] != config_digest(teacher):
        raise ValueError("Teacher configuration fingerprint is inconsistent")
    if identity["source_sha256"] != source_sha256:
        raise ValueError("Teacher and diagnostic must use the same reviewed implementation")
    if (teacher.get("phase"), teacher["setting"], teacher["variant"]) != ("main_study", "e1", "B"):
        raise ValueError("Teacher must be a validation-selected main-study e1/B mechanism")
    if (student.get("phase"), student["setting"], student["variant"]) != ("diagnostic_B_to_A", "e1", "A"):
        raise ValueError("Student must be a separate e1/A distillation diagnostic")
    if teacher["seed"] != student["seed"] or teacher.get("environment", {}) != student.get("environment", {}):
        raise ValueError("Teacher and student must share training seed and environment")
    if any(teacher.get(key) != student.get(key) for key in MATCHED_CONTROLS):
        raise ValueError("Fine-tuning and audit budgets must match the teacher study")
    if not 10000000 <= student["evaluation_seed"] < 11000000:
        raise ValueError("Reserve 10000000..10999999 for diagnostic evaluation")
    selection = json.loads(raw["frozen_selection.json"])
    if selection.get("role") != "mechanism_selection_validation_not_final_test":
        raise ValueError("Teacher selection must use validation only")
    if selection.get("passes_selection_audit") is not True:
        raise ValueError("No validated B teacher is available; preserve the failed search")
    frozen = pickle.loads(raw["frozen_selection.pkl"])
    if "mechanism_params" not in frozen:
        raise ValueError("Missing frozen teacher parameters")
    return identity, selection, frozen["mechanism_params"], raw


def load_study_initialization(config, output, source_sha256):
    """Verify both the fitted parameters and their metadata before unpickling.

    Artifact paths are relative to the study output for portable archived runs.
    Only diagnostic A studies accept a warm start; main comparisons stay random.
    """
    spec = config.get("initialization")
    if spec is None:
        return None
    if (config.get("phase"), config["setting"], config["variant"]) != ("diagnostic_B_to_A", "e1", "A"):
        raise ValueError("A fitted initialization is restricted to the B-to-A diagnostic")
    if not 10000000 <= config["evaluation_seed"] < 11000000:
        raise ValueError("Diagnostic evaluation namespace required")
    root = Path(output)
    metadata = json.loads(read_verified(root/spec["manifest"], spec["manifest_sha256"]))
    expected = {"role": "validation_selected_distilled_A_initialization", "variant": "A",
        "setting": "e1", "seed": config["seed"], "environment": config.get("environment", {}),
        "source_sha256": source_sha256, "params_sha256": spec["params_sha256"]}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Initialization identity differs from this diagnostic")
    return pickle.loads(read_verified(root/spec["params"], spec["params_sha256"]))
