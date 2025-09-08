from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

text = [
    "Молоко 2,5% жирности, литр", # Еда
    "Смартфон с экраном 6,5 дюймов", # Техника
    "Джинсы синие, размер 48", # Одежда
    "Картофель, килограммы", # Еда
    "Компьютер с экраном 30 дюймов", # Техника
    "Кофта молочная, размер S", # Одежда
    "Чипсы, граммы", # Еда
    "Ноутбук с экраном 14 дюймов", # Техника
    "Куртка зеленая, размер XXL" # Одежда
]

lable = [1,2,3,1,2,3,1,2,3] # 1 - Еда, 2 - Техника, 3 - Одежда

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=12)

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

print("New: ", pipe.predict(["Планшет с экраном 12 дюймов"])[0])
print("New: ", pipe.predict(["Куртка молочная"])[0])
print("New: ", pipe.predict(["Шорты черные, размер М"])[0])