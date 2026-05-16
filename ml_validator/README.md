# Валидатор ML-Каскада

Отдельная папка для экспериментов с ONNX-каскадом без обучения моделей.

Каскад работает как бинарный фильтр:

1. Сначала запускается `garment_detector`.
2. Если детектор одежды не нашел bbox выше `garment_conf`, возвращается `REJECT / no_garment_detected`.
3. Если одежда найдена, запускается второй детектор: `bad_classes_detector`.
4. Если второй детектор нашел плохой класс выше нужного порога, возвращается `REJECT / bad_class_detected`.
5. Если плохих классов нет, возвращается `ACCEPT / ok`.

## Структура Папки

```text
ml_validator/
  config.yaml
  requirements.txt
  README.md
  models/
    garment_detector.onnx
    bad_classes_detector.onnx
  notebooks/
    01_cascade_inference_playground.ipynb
    02_cascade_validation.ipynb
  src/
    yolo_onnx.py
    cascade.py
    class_rules.py
    yolo_dataset.py
    metrics.py
    visualization.py
  outputs/
```

## Установка

```powershell
cd C:\MY_Data\Programming\Work\UGNTU_TGBOT\ml_validator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name ml-validator --display-name "ml-validator"
jupyter notebook
```

В Jupyter нужно выбрать kernel `ml-validator`.

## Главный Конфиг

Все основные параметры лежат в `config.yaml`.

Там можно менять:

- пути к моделям;
- путь к датасету и split;
- размер входа YOLO;
- пороги confidence;
- включение/выключение NMS;
- правила плохих классов.

Важные текущие правила:

- `person` дает reject только если `confidence >= person_conf`;
- все остальные плохие классы используют `bad_class_conf`;
- `bed`, `chair`, `couch`, `dining table` специально не считаются плохими классами.

## Ноутбуки

`notebooks/01_cascade_inference_playground.ipynb`

Ноутбук для ручной проверки одной картинки или маленькой пачки картинок.

Что внутри:

- загрузка `config.yaml`;
- загрузка двух ONNX-моделей;
- запуск каскада на одной картинке;
- печать сырого JSON-ответа каскада;
- отрисовка результата через `matplotlib`;
- отрисовка predicted bbox и GT bbox на одной картинке;
- прогон маленькой пачки картинок с выводом сеткой;
- автоматический поиск и показ примеров `TP`, `TN`, `FP`, `FN`.

Цвета bbox в визуализации:

- синий bbox — предсказания `garment_detector`;
- желтый bbox — плохие классы из `bad_classes_detector`, которые прошли reject-threshold;
- зеленый bbox — GT-разметка из YOLO label-файла.

Этот ноутбук нужен, чтобы глазами проверять:

- где garment detector нашел одежду;
- где bad-class detector сработал как reject-filter;
- где GT-разметка совпадает или не совпадает с предсказаниями;
- какие картинки дают `TP`, `TN`, `FP`, `FN`.

`notebooks/02_cascade_validation.ipynb`

Ноутбук для валидации YOLO-датасета такой структуры:

```text
датасет/
  images/train
  images/val
  labels/train
  labels/val
```

Правило бинарной валидации:

- `ACCEPT` + в GT есть bbox = `TP`;
- `ACCEPT` + GT-разметка пустая = `FP`;
- `REJECT` + GT-разметка пустая = `TN`;
- `REJECT` + в GT есть bbox = `FN`.

Что внутри:

- загрузка `config.yaml`;
- загрузка каскада;
- сбор картинок и label-файлов из выбранного split;
- подсчет баланса GT: сколько positive и negative примеров;
- прогон каскада по датасету;
- расчет confusion matrix: `TP`, `TN`, `FP`, `FN`;
- расчет метрик:
  - `accuracy`;
  - `precision`;
  - `recall`;
  - `specificity`;
  - `f1`;
- подсчет причин reject:
  - `no_garment_detected`;
  - `bad_class_detected`;
- сохранение CSV-отчета по каждой картинке в `outputs/`;
- таблица ошибок для анализа `FP` и `FN`;
- визуализация нескольких ошибок с predicted bbox и GT bbox.

CSV-отчет содержит:

- путь к картинке;
- путь к label-файлу;
- есть ли GT bbox;
- решение каскада;
- reason;
- bucket: `TP`, `TN`, `FP`, `FN`;
- количество garment bbox;
- количество bad-class bbox;
- названия найденных плохих классов;
- максимальный confidence garment detector;
- максимальный confidence bad-class detector;
- общее время обработки картинки.
