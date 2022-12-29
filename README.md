# ML-проект. Классификация текстов

В проекте решается задача определения авторства русскоязычных текстов. 

В датасете собраны отрывки произведений 34 русских (российских, советских) писателей. 

Объектами в датасете являются отрывки длиной 2500 символов (плюс символы до конца предложения). Произведения в обучающем и тестовом датасетах не пересекаются.

## Данные

Данные можно скачать с помощью kaggle api отсюда:
  
<pre><code>kaggle datasets download --unzip tatianafofanova/authorstexts
</code></pre>
  

В обучающей выборке - более 97 тыс. объектов (34 автора, 182 произведения)

В тестовой выборке - более 38 тыс. объектов (34 автора, 64 произведения)

## Модель BERT

В качестве исходной предобученной модели была использована модель <code>sberbank-ai/ruBert-base</code> из библиотеки huggingface.

Дообучение модели логировалось с помощью сервиса wandb, лог доступен по <a href="https://wandb.ai/sava_ml/hw-nlp/runs/17qity9i/overview?workspace=user-sava_ml">ссылке</a>.

Код fine-tunning'а предобученной модели - файл <code>train_big_model.py</code>.

Сохраненный артефакт модели можно использовать с помощью кода:

<pre><code>import wandb
run = wandb.init()
artifact = run.use_artifact('sava_ml/hw-nlp/model_34_baseline:v1', type='model')
artifact_dir = artifact.download()
</code></pre>

Качество модели на тестовой выборке f1-score(macro) = 0.75.

## Дистилляция

Так как модель BERT очень громоздкая, то для инференса можно попробовать ее уменьшить. В проекте была сделана дистилляция модели с уменьшением количества энкодеров BERT в 2 раза. 

Веса дистиллированной модели были инициализированны весами исходной модели, затем уменьшенная модель дообучалась на 1 эпохе с применением кастомной функции потерь (кросс-энтропия плюс KL-дивергенция).

Качество модели на тестовой выборке упало до f1-score(macro) = 0.73.

Лог дистилляции находится <a href="https://wandb.ai/sava_ml/uncategorized/runs/vsy952jo/overview?workspace=user-sava_ml">здесь</a>.

Использование артефакта дистиллированной модели:

<pre><code>import wandb
run = wandb.init()
artifact = run.use_artifact('sava_ml/hw-nlp/distilled_bert:v0', type='model')
artifact_dir = artifact.download()
</code></pre>

## Flask - приложение

Инференс модели можно проверить с помощью приложения. Запустить его можно, собрав docker-контейнер:

<pre><code>docker build . -t ml_project
docker run -p 5000:5000 ml_project
</code></pre>


## файлы проекта
get_data.sh - скрипт для загрузки данных

train_big_model.py - обучение большой модели

distillation.py - дистилляция модели

app.ry - приложение flask

Dockerfile - создание образа для запуска приложения

requirements.txt - зависимости для приложения

templates/ -  директория для шаблонов (в ней единсвенный html-файл, в который рендерится предсказание модели)

stemmed_writers_rus.json - описание классов датасета
