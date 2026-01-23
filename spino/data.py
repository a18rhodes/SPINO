import os
import tarfile
from pathlib import Path


def backup_artifacts(unique_run_name: str):
    """
    Safely copies results from the Container to the Mounted Backup Drive.
    """
    backup_root = os.environ.get("SPINO_BACKUP_DIR")
    unique_run_path = Path(unique_run_name)
    unique_run_name_safe = unique_run_path.name
    run_parent = unique_run_path.parent
    if not backup_root or not Path(backup_root).exists():
        print(f"Skipping backup: SPINO_BACKUP_DIR not set or {backup_root} not found.")
        return
    backup_path = Path(backup_root)
    backup_path.mkdir(parents=True, exist_ok=True)
    tar_filename = backup_path / f"{unique_run_name_safe}.tar.gz"
    print(f"Backing up artifacts to: {tar_filename}")
    try:
        with tarfile.open(tar_filename, "w:gz") as tar:
            src_model = Path("models", unique_run_path).with_suffix(".pt")
            if src_model.exists():
                tar.add(src_model, arcname=str(Path(unique_run_name_safe, "model.pt")))
            src_figs = Path("figures", run_parent)
            for fig in src_figs.rglob(f"{unique_run_name_safe}.png"):
                relative_path = fig.relative_to(src_figs)
                tar.add(fig, arcname=str(Path(unique_run_name_safe, "figures") / relative_path))
            src_runs = Path("runs", unique_run_path)
            if src_runs.exists():
                tar.add(src_runs, arcname=str(Path(unique_run_name_safe, "runs")))
    except Exception as e:
        print(f"Backup Warning: Failed to copy some files: {e}")
