import contractions
import en_core_web_sm, nltk, string, re
from bs4 import BeautifulSoup

class Preprocessing():
    def __init__(self):
        self.nlp = en_core_web_sm.load()
        super(Preprocessing, self).__init__()

    def lowercase(self, text):
        return text.lower()

    def sentenceTokenizer(self, text):
        return nltk.sent_tokenize(text)

    def wordTokenizer(self, text):
        return nltk.word_tokenize(text)

    def tokenLemmatizer(self, tokens):
        return  ' '.join([token.lemma_ for token in list(self.nlp(tokens)) if (token.is_stop == False)])

    def removePuntuation(self, text):
        return re.sub('[%s]' % re.escape(string.punctuation), '', text)

    def removeSpecialCharacter(self, text='', removeDigits=True):
        if removeDigits:
            return re.sub(r'[^a-zA-Z\s]', ' ', str(text))
        else: 
            return re.sub(r'\W+', ' ', str(text))

    def expandContractions(self, text):
        return contractions.fix(text)

    def posTagging(self, text):
        return [{token.text:token.pos_ for token in self.nlp(text)}]

    def nameEntityExtration(self, text):
        return [{token.text:token.label_ for token in self.nlp(text).ents}]

    def removeHTMLtags(self, text):
        return BeautifulSoup(text,  features="lxml").text
    
    def removeURLs(self, text):
        urlPattern = re.compile(r'https?://\S+|www\.\S+')
        return urlPattern.sub(r'', text)

    def removeExtraSpace(self, text):
        return re.sub(' +',' ',text)

    def normalizeText(self, text, lowercase=True, removeNumberAndSpecialCharacter=True, removeDigits=False,
                            removeExtraSpace=True, expandContractions=True, 
                            removePuntuation=True, sentenceTokenizer=False, wordTokenizer=False, tokenLemmatizer=False, removeURLs=True, removeHTMLtags=True
                            ):
        try: 
            if lowercase:
                text = self.lowercase(text)
            if removeURLs:
                text = self.removeURLs(text)
            if removeHTMLtags:
                text = self.removeHTMLtags(text)
            if expandContractions:
                text = self.expandContractions(text)
            if removeNumberAndSpecialCharacter:
                text = self.removeSpecialCharacter(text=text, removeDigits=removeDigits)
            if removePuntuation:
                text = self.removePuntuation(text)
            if removeExtraSpace:
                text = self.removeExtraSpace(text)
            if tokenLemmatizer:
                text = self.tokenLemmatizer(text)
            if wordTokenizer:
                if sentenceTokenizer:
                    raise Exception ("wordTokenizer: Word Tokenizer and Sentence Tokenizer can't be called at the same time")
                text = self.wordTokenizer(text)
            if sentenceTokenizer:
                if wordTokenizer:
                    raise Exception ("sentenceTokenizer: Word Tokenizer and Sentence Tokenizer can't be called at the same time")
                else:
                    text = self.sentenceTokenizer(text)
            return text
        except Exception as e:
            return ("Exception in Preprocessing: ",e)
        
if __name__ == "__main__":
    prprocess = Preprocessing().normalizeText("ad sales boost time warner profit")
    print("prec: ", prprocess)