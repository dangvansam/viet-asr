"""
Pipeline checkpoint and state management for resumability.

This module provides checkpoint functionality to track which files have been
processed at each step, enabling the pipeline to resume from where it left off
after interruptions or failures.

Adapted from /home/samdv/data-processing-pipeline/pipeline/checkpoint.py.
Changes: replaced print() calls with loguru logger.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from loguru import logger


class PipelineCheckpoint:
    """
    Manages pipeline state and checkpointing for resumability.

    The checkpoint system tracks which files have been processed at each step,
    allowing the pipeline to skip already-processed files when resumed.

    Attributes:
        checkpoint_dir: Directory to store checkpoint files
        checkpoint_file: Path to the main checkpoint JSON file
        state: Current pipeline state dictionary
        lock: Thread lock for safe concurrent access
    """

    def __init__(self, checkpoint_dir: str):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / "pipeline_state.json"
        self.state: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

        # Load existing state if available
        self.load_state()

    def load_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Load pipeline state from checkpoint file.

        Returns:
            Dictionary containing the pipeline state
        """
        with self.lock:
            if self.checkpoint_file.exists():
                try:
                    with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                        self.state = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint file: {e}")
                    logger.warning("Starting with empty state.")
                    self.state = {}
            else:
                self.state = {}

            return self.state

    def save_state(self, state: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Save pipeline state to checkpoint file.

        Args:
            state: State dictionary to save (uses current state if None)
        """
        with self.lock:
            if state is not None:
                self.state = state

            # Add metadata
            if "_metadata" not in self.state:
                self.state["_metadata"] = {}

            self.state["_metadata"]["last_updated"] = datetime.now().isoformat()

            try:
                # Write to temporary file first, then rename (atomic operation)
                temp_file = self.checkpoint_file.with_suffix(".json.tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, ensure_ascii=False, indent=2)

                # Atomic rename
                temp_file.replace(self.checkpoint_file)
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")

    def is_processed(self, file_path: str, step: str) -> bool:
        """
        Check if a file has been processed at a given step.

        Args:
            file_path: Path (or record id) to check
            step: Pipeline step name (e.g., 'extract_audio', 'transcribe')

        Returns:
            True if file has been processed, False otherwise
        """
        with self.lock:
            return file_path in self.state.get(step, {})

    def mark_processed(
        self,
        file_path: str,
        step: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark a file as processed at a given step and save metadata.

        Args:
            file_path: Path (or record id) to mark
            step: Pipeline step name
            metadata: Optional metadata dictionary to store with the file
        """
        with self.lock:
            if step not in self.state:
                self.state[step] = {}

            self.state[step][file_path] = {
                "processed_at": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

    def mark_batch_processed(
        self,
        file_paths: list,
        step: str,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Mark multiple files as processed at once (more efficient).

        Args:
            file_paths: List of file paths / record ids to mark
            step: Pipeline step name
            metadata: Optional dict mapping file paths to their metadata
        """
        with self.lock:
            if step not in self.state:
                self.state[step] = {}

            timestamp = datetime.now().isoformat()
            for file_path in file_paths:
                file_metadata: Dict[str, Any] = {}
                if metadata and file_path in metadata:
                    file_metadata = metadata[file_path]

                self.state[step][file_path] = {
                    "processed_at": timestamp,
                    "metadata": file_metadata,
                }

    def get_processed_files(self, step: str) -> Set[str]:
        """
        Get set of all processed files for a given step.

        Args:
            step: Pipeline step name

        Returns:
            Set of file paths that have been processed
        """
        with self.lock:
            return set(self.state.get(step, {}).keys())

    def get_file_metadata(self, file_path: str, step: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific file at a given step.

        Args:
            file_path: Path to the file
            step: Pipeline step name

        Returns:
            Metadata dictionary if file was processed, None otherwise
        """
        with self.lock:
            step_data = self.state.get(step, {})
            file_data = step_data.get(file_path)
            if file_data:
                return file_data.get("metadata", {})
            return None

    def get_unprocessed_files(self, all_files: list, step: str) -> list:
        """
        Get list of files that haven't been processed at a given step.

        Args:
            all_files: List of all files / record ids to check
            step: Pipeline step name

        Returns:
            List of unprocessed file paths
        """
        processed = self.get_processed_files(step)
        return [f for f in all_files if f not in processed]

    def get_step_statistics(self, step: str) -> Dict[str, Any]:
        """
        Get statistics for a given step.

        Args:
            step: Pipeline step name

        Returns:
            Dictionary with statistics (count, earliest/latest timestamps)
        """
        with self.lock:
            step_data = self.state.get(step, {})
            if not step_data:
                return {
                    "processed_count": 0,
                    "earliest": None,
                    "latest": None,
                }

            timestamps = [
                v["processed_at"] for v in step_data.values() if "processed_at" in v
            ]
            return {
                "processed_count": len(step_data),
                "earliest": min(timestamps) if timestamps else None,
                "latest": max(timestamps) if timestamps else None,
            }

    def clear_step(self, step: str) -> None:
        """
        Clear all checkpoint data for a specific step.

        Args:
            step: Pipeline step name to clear
        """
        with self.lock:
            if step in self.state:
                del self.state[step]
                self.save_state()

    def clear_all(self) -> None:
        """Clear all checkpoint data (reset pipeline state)."""
        with self.lock:
            self.state = {}
            self.save_state()

    def get_progress_summary(self, all_steps: list) -> str:
        """
        Get a summary of progress across all steps.

        Args:
            all_steps: List of all pipeline step names

        Returns:
            Formatted string with progress summary
        """
        summary_lines = ["Pipeline Progress Summary:"]
        summary_lines.append("=" * 50)

        for step in all_steps:
            stats = self.get_step_statistics(step)
            count = stats["processed_count"]
            latest = stats["latest"] or "N/A"
            summary_lines.append(f"{step}: {count} files (last: {latest})")

        return "\n".join(summary_lines)
