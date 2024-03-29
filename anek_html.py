# сайт с анекдотами

# URL_html_veselun = r'https://веселун.рф/anekdoty/{}'
URL_html_veselun = r'https://xn--b1agaykvq.xn--p1ai/anekdoty/{}'

# URL_html = 'https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3'

''''
если ошибка urllib HTTPS request: urlopen error unknown url type: https
тогда 
I used anaconda, and I simply moved these two file to anaconda3\\DLLs and it worked.

    libcrypto-1_1-x64.dll
    libssl-1_1-x64.dll
'''''

'''
https://ru.stackoverflow.com/questions/783732/a-value-is-trying-to-be-set-on-a-copy-of-a-slice-from-a-dataframe-%D0%BD%D0%B5-%D0%BF%D0%BE%D0%B9%D0%BC%D1%83-%D0%BA%D0%B0%D0%BA
при проблемах с записью в dataframe следует избегать ][
A value is trying to be set on a copy of a slice from a DataFrame.
'''

import urllib.request
import logging
from datetime import datetime
import main_log
import time
import json
from tqdm import tqdm  # красивый прогресс-бар
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import numba
import fuzzywuzzy

try:
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

import re
import pandas as pd

def process_page_veselun(pagenum):
    with urllib.request.urlopen(URL_html_veselun.format(pagenum)) as file:
        html = file.read().decode('utf-8')
    parsed_html = BeautifulSoup(html, features="html5lib")
    quotes = parsed_html.find_all(class_='sMesVidMat2')  # получили все анекдоты
    for quote in quotes:
        # print(quote.contents[2])
        anek_list.append(quote.contents[2].prettify())


def process_all_pages_veselun():
    try:
        for i in tqdm(range(2315)):
            # for i in tqdm(range(5)):
            process_page_veselun(i)
    except:
        pass
    with open('data_veselun.json', 'w', encoding='utf-8') as file_out:
        file_out.write(json.dumps(anek_list, ensure_ascii=False, indent=4))

# безопасное превращение из одного типа в другой
# safe_cast('tst', int) # will return None
# safe_cast('tst', int, 0) # will return 0
def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def process_page_anekdotovnet(url):
    with urllib.request.urlopen(url) as file:
        html = file.read().decode('windows-1251')
    parsed_html = BeautifulSoup(html, features="html.parser")
    aneki = parsed_html.find_all(class_='anekdot')  # получили все анекдоты
    nizposta = parsed_html.find_all('table', class_='nizposta')  # получили все таблицы с оценками
    for (quote, niz) in zip(aneki, nizposta):
        # print(quote.contents[2])
        try:
            raw = niz.prettify()
            matchObj = re.findall(r'Рейтинг ([0-9.\-]+)/([0-9]+)', raw, re.M | re.I)
            ocenka = safe_cast(matchObj[0][0], float, -1)
            koll = safe_cast(matchObj[0][1], float, -1)
        except:
            ocenka = -1
            koll = -1
            print(quote.text)
            # print(raw)
        # Проверить бы Показать полностью... (до 1 мин. чтения) anekdotov.net
        # Студент 20 с небольшим лет попал под автобус, и — насмерть. Очнулся anekdotov.net
        # http://anekdotov.net/anekdot/arc/051227.html
        if (koll > 5) and (ocenka > 6.5):
            anek_list.append(quote.text)
    pagenavibig = parsed_html.find('table', class_='pagenavibig')  # навигация
    u1 = pagenavibig.find_all('a')
    return r'http://anekdotov.net' + u1[3].attrs['href'], len(aneki)


def process_all_pages_anekdotovnet(begin_url):
    sleep_time = 1
    url = begin_url
    try:
        for i in tqdm(range(15 * 365)):  # пробуем поднять весь архив за 15 лет
            urlb = url
            try:
                url, koll = process_page_anekdotovnet(url)
                if koll < 2:
                    print('На {} что-то пошло не так'.format(urlb))
                    print('Ответ {} {}'.format(url, koll))
                    break
            except Exception as e:
                print(e)
                sleep_time += 0.5
                print('Url {}'.format(url))
                time.sleep(sleep_time)
    except Exception as e:
        print(e)
        sleep_time += 0.5
        print('Url {}'.format(url))
        time.sleep(sleep_time)
    finally:
        pass
    with open('data_anekdotovnet.json', 'w', encoding='utf-8') as file_out:
        file_out.write(json.dumps(anek_list, ensure_ascii=False, indent=4))


main_log.init('anek.log', 'Anek')
print = main_log.printL
print('--------------- START -----------------')
start_time = time.time()

today = datetime.now()
todaystr = today.strftime("%Y-%m-%d %H:%M:%S.%f")
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')

anek_list = list()

# process_all_pages_anekdotovnet(r'http://anekdotov.net/anekdot/arc/201229.html')
# process_all_pages_veselun() #загрузка из интернета всех страниц

print('Чтение файла data.json')
with open('data.json', encoding='utf-8') as f:
    data = json.load(f)

#data = data[:1000]  # для скорости берем только часть
print('Число объектов {}.'.format(len(data)))


#@numba.jit(nopython=True, cache=True)
def delete_dubles(anek_df_src, vectora_idf):
    anek_dataframe = anek_df_src.copy()
    anek_dataframe = anek_dataframe.sort_values(by=['cluster'])
    # x_vec = vectora_idf.toarray()
    for i in range(len(anek_df_src) - 1):
        cell = anek_dataframe.iloc[i]
        cluster_num = cell['cluster']
        k = i + 1
        index1 = anek_dataframe.index[i]
        while (k < len(anek_dataframe)) and cluster_num == anek_dataframe.iloc[k]['cluster']:
            index2 = anek_dataframe.index[k]
            # dist =  cx(x_vec[index1], x_vec[index2])
            dist = cosine_similarity(vectora_idf[index1], vectora_idf[index2]).flatten()
            len_min = min(len(anek_dataframe.iloc[i]['Анекдот']), len(anek_dataframe.iloc[k]['Анекдот']))
            len_max = max(len(anek_dataframe.iloc[i]['Анекдот']), len(anek_dataframe.iloc[k]['Анекдот']))
            popravka = 0.00122 * len_min - 0.0007 * len_max - 0.01629 * len_min / len_max - 0.00055 * (
                        len_min / len_max) ** 2 - 0.04066 * dist * len_min / len_max
            if (dist>0.4 and 0.49 < dist + popravka < 0.51) and (
                    len_min / len_max > 0.5):
                print('.')
                print(cell['Анекдот'])
                print(' - - - - - - - - - - ')
                print(anek_dataframe.iloc[k]['Анекдот'])
                print('Совпадение {} lmin={}, lmax={}, mi/ma={}   popravka {}'.format(dist, len_min, len_max,
                                                                                      len_min / len_max, popravka))
            if dist > 0.4 and dist + popravka > 0.55 and (
                    len_min / len_max > 0.8):
                record= anek_dataframe.iloc[i].copy()
                record['Анекдот']='deleted'
                record['cluster'] = 0
                anek_dataframe.iloc[i] = record
                # print('Совпадение {} index1={}  index2={}'.format(dist, index1, index2))
                break
            if k - i > 110:
                break  # видно кластер большой, будет всё тормозить
            k += 1
    return anek_dataframe


def recursive_split_big_to_small(data : list, idx_low, idx_high):
    # делим условно пополам
    vectorizer = TfidfVectorizer(analyzer='char', dtype=np.float32, ngram_range=(5, 7), max_features=20000)
    if len(data)>500:
        vectorizer.max_df = 0.95
    # print(vectorizer.get_feature_names()) # распечатаем слоги, на основе которых работаем
    X = vectorizer.fit_transform(data)
    if len(data)<15:
        # удалим все дубли, вернем результат
        anek_df = pd.DataFrame( data, columns=['Анекдот'])
        anek_df['cluster'] = round( (idx_high+idx_low) / 2)
        anek_df=delete_dubles(anek_df, X)
        pbar.update(len(data))
        return  anek_df

    model = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=3)
    model.fit(X)
    labels = model.labels_
    anek_df = pd.DataFrame(list(zip(labels, data)), columns=['cluster', 'Анекдот'])

    data0 = anek_df[anek_df['cluster']==0]
    data0 = data0['Анекдот'].tolist()
    data1 = anek_df[anek_df['cluster']==1]
    data1 = data1['Анекдот'].tolist()

    df0 = recursive_split_big_to_small(data0, idx_low, round((idx_low + idx_high) / 2) - 1)
    df1 = recursive_split_big_to_small(data1, round((idx_low + idx_high) / 2) + 1, idx_high)
    res= pd.concat( [df0, df1], ignore_index=True)
    return res

def recursive_split_to_small(data : list, idx_offset: int):
    if len(data)<1:
        print('Error data')
        return
    # делим на большое число кластеров
    cluster_count = round( len(data) / 30) # будем считать, что средний размер кластера должен быть 30
    if cluster_count<2:
        cluster_count = 2
    if cluster_count>200:
        cluster_count = 200
    vectorizer = TfidfVectorizer(analyzer='char', dtype=np.float32, ngram_range=(5, 7), max_features=20000)
    if len(data)>500:
        vectorizer.max_df = 0.95
    # print(vectorizer.get_feature_names()) # распечатаем слоги, на основе которых работаем
    X = vectorizer.fit_transform(data)
    if len(data)<15:
        # удалим все дубли, вернем результат
        anek_df = pd.DataFrame( data, columns=['Анекдот'])
        anek_df['cluster'] = idx_offset
        anek_df=delete_dubles(anek_df, X)
        pbar.update(len(data))
        return  anek_df, idx_offset + 1

    model = KMeans(n_clusters=cluster_count, init='k-means++', max_iter=400, n_init=3)
    model.fit(X)
    labels = model.labels_
    anek_df = pd.DataFrame(list(zip(labels, data)), columns=['cluster', 'Анекдот'])

    next_clust_number =idx_offset
    data0 = anek_df[anek_df['cluster'] == 0]
    data0 = data0['Анекдот'].tolist()
    anek_res_df, next_clust_number = recursive_split_to_small(data0, next_clust_number)

    for i in range(1, cluster_count):
        data0 = anek_df[anek_df['cluster']==i]
        data0 = data0['Анекдот'].tolist()
        if len(data0) < 1:
            continue
        part_i, next_clust_number = recursive_split_to_small(data0, next_clust_number  )
        anek_res_df = pd.concat([anek_res_df, part_i], ignore_index=True)

    return anek_res_df, next_clust_number

from fuzzywuzzy import fuzz
def calculate_similarity_of_list_neighbor(data:list):
    #str1 = 'Та кокетливо отвечает: — Мне ближе к тридцати, чем к двадцати.'
    #str2 = '— Мне ближе к тридцати, чем к двадцати пяти.'
    #print(fuzz.token_sort_ratio(str1,str2 ))
    # https://www.datacamp.com/community/tutorials/fuzzy-string-python
    result=0
    count=0
    for i in range(len(data)-4):
        result += fuzz.token_sort_ratio(data[i],data[i+1] )
        result += fuzz.token_sort_ratio(data[i],data[i+2] )
        result += fuzz.token_sort_ratio(data[i],data[i+3] )
        result += fuzz.token_sort_ratio(data[i],data[i+4] )
        count += 4
    return result/count

def calculate_similarity_of_list_random(data:list):
    #str1 = 'Та кокетливо отвечает: — Мне ближе к тридцати, чем к двадцати.'
    #str2 = '— Мне ближе к тридцати, чем к двадцати пяти.'
    #print(fuzz.token_sort_ratio(str1,str2 ))
    # https://www.datacamp.com/community/tutorials/fuzzy-string-python
    import random
    result=0
    count=0
    for i in range(len(data)-4):
        i2 = random.randrange(len(data))
        result += fuzz.token_sort_ratio(data[i],data[i2] )
        count += 1
    return result/count

def copmape_process_():
    # сравним насколько хорошо разбиваются на кластеры
    # кластеры должны быть максимально похожие внутри. Желательно, что бы и соседние были похожими
    global pbar
    print('Начинаем сравнение двух подходов recursive_split_to_small и recursive_split_big_to_small')
    var1 = list()
    var2 = list()
    for i in range(4):
        pbar = tqdm(total=len(data))
        anek_df, _ = recursive_split_to_small(data, 1)
        pbar.close()
        a = calculate_similarity_of_list_neighbor(anek_df.sort_values(by=['cluster'])['Анекдот'].to_list())
        var1.append(a)
        print('recursive_split_to_small {}'.format(a))

    for i in range(4):
        pbar = tqdm(total=len(data))
        anek_df = recursive_split_big_to_small(data, 100_000,999_000)
        pbar.close()
        a = calculate_similarity_of_list_neighbor(anek_df.sort_values(by=['cluster'])['Анекдот'].to_list())
        var2.append(a)
        print('recursive_split_big_to_small {}'.format(a))
    print('Результаты проверки (больше - лучше)')
    print('recursive_split_to_small')
    print(sorted( var1))
    print('recursive_split_big_to_small')
    print(sorted( var2))

if input('Нужно сравнить два подхода to_small и big_to_small? [Y/N] ').lower() in ('y', 'yes', 'да', '1'):
    copmape_process_()

pbar = tqdm(total=len(data))
anek_df, _ = recursive_split_to_small(data , 1)
pbar.close()

print('Запись результата anek-klasters.csv')
anek_df.sort_values(by=['cluster']).to_csv('anek-klasters.csv', index=False)

with open('data_clean_clustered.json', 'w', encoding='utf-8') as f:
    f.write(anek_df.sort_values(by=['cluster'])['Анекдот'].to_json(force_ascii=False, indent=2, orient='values'))

a1=calculate_similarity_of_list_neighbor(anek_df.sort_values(by=['cluster'])['Анекдот'].to_list())
a2 = calculate_similarity_of_list_random(anek_df.sort_values(by=['cluster'])['Анекдот'].to_list())
print('Похожесть осортированных {}; случайных {}'.format(a1, a2))

logging.warning('--- run time {:.2f} seconds ---'.format(time.time() - start_time))
