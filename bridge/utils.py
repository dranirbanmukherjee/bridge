"""Utility classes and functions for the BRIDGE pipeline."""

import time


class Timer:
    """Simple timer for tracking operation durations."""

    def __init__(self):
        self.times = {}
        self.start_time = None
        self.current_step = None

    def start(self, step_name: str):
        """Start timing a step."""
        self.current_step = step_name
        self.start_time = time.time()

    def stop(self):
        """Stop timing the current step and record duration."""
        if self.current_step and self.start_time:
            elapsed = time.time() - self.start_time
            self.times[self.current_step] = elapsed
            print(f"  ⏱ {self.current_step}: {self._format_time(elapsed)}")
            self.current_step = None
            self.start_time = None

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.0f}s"

    def summary(self):
        """Print timing summary."""
        total = sum(self.times.values())
        print("\n" + "=" * 70)
        print("TIMING SUMMARY")
        print("=" * 70)
        for step, elapsed in self.times.items():
            pct = (elapsed / total * 100) if total > 0 else 0
            print(f"  {step:40s} {self._format_time(elapsed):>12s} ({pct:5.1f}%)")
        print("-" * 70)
        print(f"  {'TOTAL':40s} {self._format_time(total):>12s}")
        print("=" * 70)
