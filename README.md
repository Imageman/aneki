# aneki
Маленький проект по автоматическому разбиению на кластеры кучи коротких анекдотов. 

anek_html.py содержит функции process_all_pages_anekdotovnet и process_all_pages_veselun для скачивания анекдотов с двух сайтов. После этого они записываются в json, который нужно вручную очистить от разного мусора (к примеру от строк вида '</br>').

Анекдоты записать в data.json (UTF-8 без BOM). При помощи функции recursive_split мы разбиваем тексты на примерно одинаковые кластера.
