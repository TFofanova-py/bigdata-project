from flask import Flask
import argparse
from flask import request, make_response, render_template, redirect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import softmax, argsort
import json

# download model
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
model = AutoModelForSequenceClassification.from_pretrained("artifacts/model_34_baseline:v1",
                                                           return_dict=True, num_labels=34,
                                                           ignore_mismatched_sizes=True)

# open writers dict
with open("stemmed_writers_rus.json") as f:
    writers_dict = json.load(f)["writers"]

app = Flask("text-classification")


def make_predict(text):
    tokenized_text = tokenizer(text, truncation=True, padding="max_length", max_length=512,
                               return_tensors="pt")
    logits = model(tokenized_text["input_ids"], tokenized_text["attention_mask"]).logits

    # получим топ-3 вероятностей авторов
    predictions = softmax(logits, dim=-1)
    top_3_idx = argsort(predictions.squeeze(), descending=True)[:3].numpy().tolist()
    top_3_pairs = [(writers_dict[model.config.id2label[idx]]["name"],
                    round(predictions[0, idx].item(), 5)) for idx in top_3_idx]
    return top_3_pairs


@app.route('/')
def hello():
    form = """
    <form method = "POST" action="/forward">
         Напишите текст на русском языке (не более 2000 символов):
         <br>
         <textarea rows = "15" cols = "100" name = "text">
         </textarea>
         <br>
         <input type = "submit" name="submit" value = "Отправить" />
      </form>
    """
    return form


@app.route("/forward", methods=["GET", "POST"])
def send_text():
    if request.method == "GET":
        return redirect("/", 302)

    elif request.method == 'POST':
        input_text = request.form["text"].strip()

        if input_text != "":
            try:
                # отправим текст в модель
                top_3_preds = make_predict(input_text)

                return render_template('view_response.html', text=input_text, prediction=top_3_preds)

            except Exception as e:
                response = make_response(f"{e}, Модель не смогла сделать предсказание", 400)
        else:
            response = make_response("Вы забыли написать текст", 400)

        return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5000)

    args = parser.parse_args()

    # run app
    app.run(host=args.host, port=args.port)
