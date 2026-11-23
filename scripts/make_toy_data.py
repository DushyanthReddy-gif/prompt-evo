"""Generate the toy datasets (deterministic). Usage: python -m scripts.make_toy_data"""
from tests.toy_tasks import write_all

if __name__ == "__main__":
    write_all("data", seed=0)
    print("wrote data/{classification,math,qa}_{dev,test}.jsonl")
