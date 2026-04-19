import textacy.preprocessing as pp
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


print(pp.__version__)
print(word_tokenize("This is a test."))