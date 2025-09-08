from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

text = [
    "Я обожаю этот день", # радость
    "Меня все бесит", # злость
    "Сегодня будет обычный день", # нейтрально
    "Мой очень хороший день", # радость
    "Я все ненавижу", # злость
    "Опять будет скучный день", # нейтрально
    "Я наслаждаюсь этим днём", # радость
    "Дерьмо случается", # злость
    "Как всегда нормально" # нейтрально
]

lable = [1,2,3,1,2,3,1,2,3] # 1 - радость, 2 - злость, 3 - нейтрально

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=52)

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

print(f"Точность: {accuracy_score(y_test,y_pred)}")

print("New: ", pipe.predict(["Наконец выходной, но будет рутина"])[0])
print("New: ", pipe.predict(["Как же все надоело"])[0])
print("New: ", pipe.predict(["Опять рутина"])[0])