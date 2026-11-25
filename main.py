"""Raqib prototype: command-line entry point.

Usage:
    python main.py preprocess
    python main.py dialect
    python main.py split
    python main.py evaluate
    python main.py train                         # needs GPU libraries, see README
    python main.py search "الطلب ما وصل" --rerank
"""
import argparse

from src.data import group_split, load_data
from src.dialect import detect_dialect, segment
from src.evaluation import bootstrap_ci, error_taxonomy, sliced_accuracy
from src.preprocessing import normalize_arabic, preprocess

# Illustrative values only (NOT measured results of a trained model).
DEMO_RECORDS = [
    {"language": "Arabic", "dialect": "Saudi White", "length": "Short", "correct": 1},
    {"language": "Arabic", "dialect": "Saudi White", "length": "Short", "correct": 1},
    {"language": "Arabic", "dialect": "Saudi White", "length": "Long", "correct": 1},
    {"language": "Arabic", "dialect": "Saudi White", "length": "Long", "correct": 0},
    {"language": "Arabic", "dialect": "Lebanese", "length": "Short", "correct": 0},
    {"language": "Arabic", "dialect": "Lebanese", "length": "Short", "correct": 1},
    {"language": "Arabic", "dialect": "Lebanese", "length": "Long", "correct": 0},
    {"language": "English", "dialect": "Standard", "length": "Short", "correct": 1},
    {"language": "English", "dialect": "Standard", "length": "Short", "correct": 1},
    {"language": "English", "dialect": "Standard", "length": "Long", "correct": 1},
    {"language": "English", "dialect": "Standard", "length": "Long", "correct": 0},
    {"language": "English", "dialect": "Standard", "length": "Long", "correct": 1},
]
DEMO_Y_TRUE = [1,0,1,1,0,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0]
DEMO_Y_PRED = [1,0,1,0,0,1,1,1,1,0,1,1,0,1,1,1,0,1,0,0]
DEMO_ERRORS = [
    ("المندوب ما رد علي من الصبح", "delivery", "service", "Topic overlap"),
    ("بدي ارجع المنتج هلق", "quality", "payment", "Dialect"),
    ("كيف أحجز تذكرة؟", "out_of_scope", "delivery", "Out-of-scope"),
    ("الطلب وصل بس ناقص قطعة", "quality", "delivery", "Multi-issue"),
    ("وين فلوسي اللي رجعتوها؟", "payment", "service", "Ambiguous wording"),
]


def cmd_preprocess(_):
    samples = [
        "  الطلب    تأخر   أسبوعين  تواصلوا على 0551234567 أو email@example.com  ",
        "  My   order is    late, call +966501112233  ",
    ]
    for text in samples:
        print("Before:", repr(text))
        print("After :", preprocess(text))
        print("Normalised:", normalize_arabic(preprocess(text)), "\n")


def cmd_dialect(_):
    for text in ["وش الأخبار اليوم؟", "بدي ارجع المنتج هلق", "الطلب تأخر كثيرا"]:
        print(f"{text:28} → {detect_dialect(text)}")
    print()
    for word in ["وبالرياض", "للمدرسة", "بطاقتي"]:
        print(f"{word:10} → {' + '.join(segment(word))}")


def cmd_split(args):
    df, topics, label_map = load_data(args.data)
    train_df, test_df = group_split(df, args.test_size, args.seed)
    print("Labels:", label_map)
    print("Train:", len(train_df), "| Test:", len(test_df))
    print("Train topics:", sorted(train_df.topic.unique()))
    print("Test topics :", sorted(test_df.topic.unique()))
    print("Shared customers: none")


def cmd_evaluate(_):
    print("(Illustrative demo data, not a real model's results)\n")
    for col, table in sliced_accuracy(DEMO_RECORDS).items():
        print(f"By {col}:\n{table}\n")
    low, high = bootstrap_ci(DEMO_Y_TRUE, DEMO_Y_PRED)
    print(f"95% CI for accuracy: {low:.2f} - {high:.2f}")
    print("Error types:", dict(error_taxonomy(DEMO_ERRORS)))


def cmd_train(args):
    from src.classifier import train_classifier

    df, topics, _ = load_data(args.data)
    train_df, test_df = group_split(df, args.test_size, args.seed)
    _, _, metrics = train_classifier(train_df, test_df, topics, epochs=args.epochs)
    print(metrics)


def cmd_search(args):
    from src.search import ComplaintSearch

    df, _, _ = load_data(args.data)
    searcher = ComplaintSearch(df["text"])
    results = searcher.rerank(args.query) if args.rerank else searcher.search(args.query)
    for text, score in results:
        print(f"{score:.2f} → {text}")


def main():
    parser = argparse.ArgumentParser(description="Raqib complaints NLP prototype")
    parser.add_argument("command", choices=["preprocess", "dialect", "split",
                                            "evaluate", "train", "search"])
    parser.add_argument("query", nargs="?", default="الطلب ما وصل")
    parser.add_argument("--data", default="data/complaints.csv")
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()
    {
        "preprocess": cmd_preprocess, "dialect": cmd_dialect, "split": cmd_split,
        "evaluate": cmd_evaluate, "train": cmd_train, "search": cmd_search,
    }[args.command](args)


if __name__ == "__main__":
    main()
