from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata


sentence = 'i hate you which piss me off'

sid = SentimentIntensityAnalyzer()
ss = sid.polarity_scores(sentence)
print(ss)

