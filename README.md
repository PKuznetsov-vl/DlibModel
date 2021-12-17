# DlibModel
68 landmarks task

https://drive.google.com/drive/folders/1InYhb6IOOXMs1cBs9Nx9KBFr7s_V6m9O?usp=sharing  - model here

## Настройка окружения для Ubuntu

> sudo apt-get install build-essential cmake pkg-config
> 
> sudo apt-get install libx11-dev libatlas-base-dev
> 
> sudo apt-get install libgtk-3-dev libboost-python-dev

Перейти в папку с проектом 

> python -m pip install -r requirements.txt

# 

# Запуск обучения и тестирования

В файле `config.json` указать

> {
> 
> "directory_for_train" : "Путь к датасету для обучения"
> 
>  "faces_folder_for_test": "Путь к датасету для тестирования''
> 
> "predictor": "путь к модели", 
> 
>  "xml": "путь где будет создан xml файл или где он лежит", 
>  
>  "predictor_output": "куда сохранится результат работы модели",
>   
>  "create_data_val": false, - этапы работы программы ,создать данные для обучения
>  
>  "train_model_val" : false, - тренировать модель
>  
>  "create_predictor_val" : true - протестировать модель
> 
> "orig_data_path": "Эталонные данные для тестирования", 
> 
> "predictor_data_path": "какой из результатов работы модели выбрать, 
> 
> "graph_output_path": "куда сохранится график" 
> }   



# Запуск скрпита

> python3 main.py
