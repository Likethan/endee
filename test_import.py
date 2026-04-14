try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("TfidfVectorizer imported OK")
except Exception as e:
    print("Error importing TfidfVectorizer:", repr(e))

try:
    from sklearn.metrics.pairwise import cosine_similarity
    print("cosine_similarity imported OK")
except Exception as e:
    print("Error importing cosine_similarity:", repr(e))

print("Flask version check:")
import flask
print(flask.__version__)

print("All imports done")
