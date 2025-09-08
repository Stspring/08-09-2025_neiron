from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

text = [
    "Купи афон за 100 рублей",              # Спам
    "Встреча в офисе в 10",                 # НЕ спам
    "Вы выиграли миллион, пришлите деняк",  # Спам
    "Отчет: продажи за квартал 2025",       # НЕ спам
    "Срочно пазвани сечас и палучи прис",   # Спам
    "Напоминаем про аплату щета"            # НЕ спам
]

lable = [1,0,1,0,1,0]    # 1 - спам, 0 - НЕ спам

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=42)

print(text_train)
print(text_test)
print(y_train)
print(y_test)

pipe = make_pipeline(
    CountVectorizer(),  
    MultinomialNB()     
)
pipe.fit(text_train, y_train)
y_pred = pipe.predict(text_test)

print(f"Accuracy: {accuracy_score(y_test,y_pred)}")

print("New: ", pipe.predict(["Срочно пазвани сечас и палучи прис"])[0])
print("New: ", pipe.predict(["Добрый вечер! Встреча завтра в 12, пришлите деняк "])[0])