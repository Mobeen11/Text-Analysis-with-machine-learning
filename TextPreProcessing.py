import contractions
import en_core_web_sm, nltk, string

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
        return dict((ord(punct), None) for punct in string.punctuation)

    def removeSpecialCharacter(self, removeDigits=True, text):
        if removeDigits:
            return text.replace('[^a-zA-Z]', '')
        else: 
            return text.replace('[^a-zA-Z0-9\s]', '')

    def expandContractions(self, text):
        return contractions.fix(text)

    def posTagging(self, text):
        return [{token.text:token.pos_ for token in self.nlp(text)}]

    def nameEntityExtration(self, text):
        return [{token.text:token.label_ for token in self.nlp(text).ents}]

    def normalizeText(self, lowercase=True, removeNumberAndSpecialCharacter=True,
                            posTagging=True, nameEntityExtration=True, expandContractions=True, 
                            removePuntuation=True, sentenceTokenizer=True, wordTokenizer=True, tokenLemmatizer=True
                            ):
        
        return 