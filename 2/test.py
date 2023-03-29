import pandas
import numpy as np

# ---------------- Ładowanie danych i oddzielanie zbioru treningowego od testowego ------

try:
    full_dataset = pandas.read_csv('spam_emails.csv', encoding='utf-8')      # wczytaj dane z pliku CSV
except:
    import s3fs
    full_dataset = pandas.read_csv("https://dwisniewski-put-pjn.s3.eu-north-1.amazonaws.com/spam_emails.csv")
full_dataset['label_num'] = full_dataset.label.map({'ham':0, 'spam':1})  # ponieważ nazwy kategorii zapisane są z użyciem stringów: "ham"/"spam", wykonujemy mapowanie tych wartości na liczby, co będzie potrzebne do wykonania klasyfikacji. 

np.random.seed(0)                                       # ustaw seed na 0, aby zapewnić powtarzalność eksperymentu
train_indices = np.random.rand(len(full_dataset)) < 0.7 # wylosuj 70% danych, które stworzą zbiór treningowy. train_indices, to wektor o długości liczności wczytanego zbioru danych, w którym każda pozycja (przykład) może przyjąć dwie wartości: 1.0 - wybierz do zbioru treningowego; 0.0 - wybierz do zbioru testowego

train = full_dataset[train_indices] # wybierz zbior treningowy (70%)
test = full_dataset[~train_indices] # wybierz zbiór testowy (dopełnienie treningowego - 30%)



# ---------------- Wyświetlanie statystyk -----------------


print("Elementów w zbiorze treningowym: {train}, testowym: {test}".format(
    train=len(train), test=len(test)
))

print("\n\nLiczność klas w zbiorze treningowym: ")
print(train.label.value_counts())  # wyświetl rozkład etykiet w kolumnie "label"

print("\n\nLiczność klas w zbiorze testowym: ")
print(test.label.value_counts())   # wyświetl rozkład etykiet w kolumnie "label"



full_dataset.head()

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(train['text']) # stwórz macierz liczbową z danych. W wierszach mamy kolejne dokumenty, w kolumnach kolejne pola wektora cech odpowiadające unikalnym słowom (bag of words)
X_test_counts = vectorizer.transform(test['text'])       # analogicznie jak wyżej - dla zbioru testowego.

print("Rozmiar stworzonej macierzy: {x}".format(x=X_train_counts.shape)) # wyświetl rozmiar macierzy. Pierwsze pole - liczba dokumentów, drugie - liczba cech (stała dla wszystkich dokumentów)
print("Liczba dokumentów: {x}".format(x=X_train_counts.shape[0]))
print("Rozmiar wektora bag-of-words {x}".format(x=X_train_counts.shape[1]))

count_tokens = 0   # tu zapisz liczbę wszystkich tokenów w macierzy
count_nonzero = 0  # tu zapisz ilość elementów niezerowych w macierzy
count_all = X_train_counts.shape[0] * X_train_counts.shape[1]      # tu zapisz ilość komórek w macierzy (ilość wierszy * ilość kolumn, rozważ użycie pola 'shape' na macierzy X_train_counts)

cx = X_train_counts.tocoo()
for doc_id, word_id, count in zip(cx.row, cx.col, cx.data):    #iteracja po elementach niezerowych
    count_tokens += count
    count_nonzero += 1

print("W datasecie znajduje się: {tokens} tokenów. Macierz posiada {nonzero_percent}% elementów niezerowych".format(
    tokens=count_tokens,
    nonzero_percent = round(100.0*count_nonzero/count_all, 3)
))