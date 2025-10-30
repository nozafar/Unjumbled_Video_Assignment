import time, json, os
from datetime import datetime

def now_ts():
    return datetime.utcnow().isoformat() + "Z"

class Timer:
    def __init__(self):
        self.records = {}
    def start(self, name):
        self.records[name] = {'start': time.time(), 'end': None}
    def stop(self, name):
        self.records[name]['end'] = time.time()
    def elapsed(self, name):
        r = self.records.get(name)
        if not r or r['end'] is None:
            return None
        return r['end'] - r['start']
    def to_json(self, path='execution_time.json'):
        out = {k: {'start': v['start'], 'end': v['end'], 'elapsed_seconds': (v['end']-v['start']) if v['end'] else None} for k,v in self.records.items()}
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
    def print_summary(self):
        for k,v in self.records.items():
            el = (v['end'] - v['start']) if v['end'] else None
            print(f"{k}: {el:.3f}s" if el else f"{k}: running")
