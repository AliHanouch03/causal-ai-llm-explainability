import json
from pathlib import Path


def load_dag(filepath):
    """Load a single DAG from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def load_all_dags(dags_dir=None):
    """Load all DAG JSON files from the dags directory."""
    if dags_dir is None:
        # Resolve path relative to this file's location
        project_root = Path(__file__).parent.parent
        dags_path = project_root / "data" / "dags"
    else:
        dags_path = Path(dags_dir)
    
    dags = []
    for filepath in sorted(dags_path.glob("*.json")):
        dags.append(load_dag(filepath))
    return dags


def dag_to_text(dag):
    """Convert a DAG's edges into a human-readable arrow format."""
    edges = dag["edges"]
    return "\n".join(f"{cause} → {effect}" for cause, effect in edges)


def dag_summary(dag):
    """Print a summary of a DAG for debugging purposes"""
    print(f"ID: {dag['id']}")
    print(f"Level: {dag['level']} | Pattern: {dag['pattern']}")
    print(f"Nodes: {', '.join(dag['nodes'])}")
    print(f"Edges:\n{dag_to_text(dag)}")
    print("-" * 60)