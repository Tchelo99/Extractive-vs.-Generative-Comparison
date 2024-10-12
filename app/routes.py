from flask import Blueprint, render_template, request, jsonify
from inference.t5_inference import load_t5_model, t5_inference
from inference.bert_inference import load_bert_model, bert_inference

main_bp = Blueprint("main", __name__)

t5_model, t5_tokenizer = load_t5_model()
bert_model, bert_tokenizer = load_bert_model()


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/qa", methods=["POST"])
def qa():
    data = request.json
    context = data.get("context")
    question = data.get("question")
    model_type = data.get("model", "t5")

    if model_type == "t5":
        answer = t5_inference(t5_model, t5_tokenizer, context, question)
    else:
        answer = bert_inference(bert_model, bert_tokenizer, context, question)

    return jsonify({"answer": answer})
