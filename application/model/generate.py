import dill


# Load source and destination field
with open("./data/ARTICLE.Field", "rb")as f:
    ARTICLE = dill.load(f)
with open("./data/SUMMARY.Field", "rb")as f:
    SUMMARY = dill.load(f)
