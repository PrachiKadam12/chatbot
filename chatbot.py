import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("conv.csv")

print(" Welcome to Prachi's Personal Chatbot ")
print("Type 'q' to quit\n")

while True:
    qts = input("You: ").strip().lower()

    if qts == "q":
        print("Bot: Bye 👋 Have a beautiful day!")
        break

    texts = [qts] + data["question"].str.lower().tolist()

    cv = CountVectorizer()
    vector = cv.fit_transform(texts)

    cs = cosine_similarity(vector)
    score = cs[0][1:]
    data["score"] = score * 100

    result = data.sort_values(by="score", ascending=False)
    result = result[result.score > 10]

    if len(result) == 0:
        print("Bot: Sorry 😔 I don't understand.")
    else:
        print("Bot:", result.head(1)["answer"].values[0])