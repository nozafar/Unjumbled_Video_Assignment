import time
import json

class Timer:
    def __init__(self):
        self.times = {}
        self.active = {}

    def start(self, name):
        self.active[name] = time.time()

    def stop(self, name):
        if name not in self.active:
            print(f"[Timer] WARNING: Timer '{name}' was never started")
            return

        elapsed = time.time() - self.active[name]
        self.times[name] = elapsed
        del self.active[name]

    def print_summary(self):
        print("\n Execution Time Summary:")
        for k, v in self.times.items():
            print(f" - {k}: {v:.3f}s")

    def to_json(self, filename="runtime.json"):
        with open(filename, "w") as f:
            json.dump(self.times, f, indent=4)
