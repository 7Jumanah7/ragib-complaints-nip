"""Raqib demo: type a complaint, get its topic, dialect and similar complaints.

Run:  python app.py            (opens at http://127.0.0.1:7860)
      python app.py --share    (public link, useful in Google Colab)
"""
import argparse

import gradio as gr

from src.data import load_data
from src.dialect import detect_dialect
from src.predict import DemoModel
from src.preprocessing import preprocess

TOPIC_AR = {
    "delivery": "توصيل",
    "payment": "دفع",
    "quality": "جودة المنتج",
    "service": "خدمة العملاء",
}

df, _, _ = load_data("data/complaints.csv")
model = DemoModel(df)


def analyze(text):
    if not text or not text.strip():
        return "", "", "", ""
    masked = preprocess(text)
    topic, conf = model.predict(masked)
    dialect = detect_dialect(masked)
    similar = "\n".join(
        f"{i}. {t}  [{TOPIC_AR[tp]}]  ({s:.2f})"
        for i, (t, tp, s) in enumerate(model.similar(masked), start=1)
    )
    return masked, f"{TOPIC_AR[topic]} ({topic}) — {conf:.0%}", dialect, similar


EXAMPLES = [
    ["الطلب ما وصل لين الحين وتواصلوا معي على 0551234567"],
    ["انخصم المبلغ مرتين من حسابي"],
    ["استلمت المنتج وهو متكسر"],
    ["ما أحد يرد على اتصالاتي"],
    ["بدي ارجع المنتج هلق"],
]

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="الشكوى", lines=3, rtl=True,
                      placeholder="اكتبي شكوى العميل هنا..."),
    outputs=[
        gr.Textbox(label="النص بعد إخفاء البيانات الحساسة", rtl=True),
        gr.Textbox(label="موضوع الشكوى (الثقة)", rtl=True),
        gr.Textbox(label="اللهجة (فحص بالكلمات)"),
        gr.Textbox(label="شكاوى مشابهة", lines=4, rtl=True),
    ],
    title="Raqib: تحليل شكاوى العملاء",
    description=("نسخة تجريبية خفيفة (TF-IDF) مدرّبة على 16 شكوى فقط، "
                 "فالنتايج تقريبية وتوضح فكرة المشروع وليست نموذجًا نهائيًا."),
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    demo.launch(share=parser.parse_args().share)
