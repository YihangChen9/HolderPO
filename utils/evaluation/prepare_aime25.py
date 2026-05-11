"""Build datasets/evaluation_suite_v2 = existing eval suite + aime25.

Source: yentinglin/aime_2025 (HF), 30 problems, fields already match {problem, answer}.
"""
from datasets import DatasetDict, load_dataset, load_from_disk

SUITE_IN = "datasets/evaluation_suite"
SUITE_OUT = "datasets/evaluation_suite_v2"


def main():
    aime25 = load_dataset("yentinglin/aime_2025", split="train")
    aime25 = aime25.select_columns(["problem", "answer"])
    suite = load_from_disk(SUITE_IN)
    new_suite = DatasetDict({**{k: v for k, v in suite.items()}, "aime25": aime25})
    new_suite.save_to_disk(SUITE_OUT)
    print(f"saved {SUITE_OUT}; splits: {list(new_suite.keys())}")


if __name__ == "__main__":
    main()
