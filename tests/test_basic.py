import numpy as np

from src.data import group_split, load_data
from src.dialect import detect_dialect, segment
from src.evaluation import bootstrap_ci, error_taxonomy, sliced_accuracy
from src.preprocessing import clean_spaces, mask_pii, normalize_arabic, preprocess


def test_clean_spaces():
    assert clean_spaces("  الخدمة    ممتازة  ") == "الخدمة ممتازة"


def test_normalize_arabic():
    assert normalize_arabic("أريد المدرسة") == "اريد المدرسه"


def test_mask_pii():
    out = mask_pii("call 0551234567 or +966501112233 or a.b@test.com")
    assert out == "call [PHONE] or [PHONE] or [EMAIL]"


def test_mask_pii_keeps_other_numbers():
    assert mask_pii("طلب رقم 12345") == "طلب رقم 12345"


def test_preprocess_pipeline():
    assert preprocess("  تواصل   0551234567 ") == "تواصل [PHONE]"


def test_dialect():
    assert detect_dialect("وش الأخبار اليوم؟") == "Saudi White"
    assert detect_dialect("بدي ارجع المنتج هلق") == "Lebanese"
    assert detect_dialect("الطلب تأخر كثيرا") == "Unknown"


def test_segment():
    assert segment("وبالرياض") == ["و", "ب", "ال", "رياض"]
    assert segment("للمدرسة") == ["ل", "ال", "مدرسة"]


def test_split_has_no_shared_customers_and_all_topics():
    df, topics, _ = load_data("data/complaints.csv")
    train_df, test_df = group_split(df, 0.5, 7)
    assert not set(train_df.customer_id) & set(test_df.customer_id)
    assert set(train_df.topic) == set(topics) == set(test_df.topic)


def test_bootstrap_ci_bounds():
    y = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    low, high = bootstrap_ci(y, y)
    assert low == high == 1.0


def test_sliced_and_taxonomy():
    records = [{"language": "Arabic", "dialect": "A", "length": "Short", "correct": 1},
               {"language": "Arabic", "dialect": "A", "length": "Long", "correct": 0}]
    assert sliced_accuracy(records)["language"].loc["Arabic", "accuracy"] == 0.5
    errs = [("x", "a", "b", "Dialect"), ("y", "a", "a", "-")]
    assert dict(error_taxonomy(errs)) == {"Dialect": 1}


def test_demo_model_predicts_known_complaint():
    from src.predict import DemoModel

    df, _, _ = load_data("data/complaints.csv")
    model = DemoModel(df)
    topic, conf = model.predict("انسحب المبلغ مرتين من بطاقتي")
    assert topic == "payment" and 0 < conf <= 1
    assert len(model.similar("المنتج وصل مكسور", k=3)) == 3
