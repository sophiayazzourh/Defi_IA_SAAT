import re
import unicodedata
import nltk
from tqdm import tqdm
from string import digits
from bs4 import BeautifulSoup #Nettoyage d'HTML

digits_list = digits

### Création d'une classe qui propose différent processus de nettoyage de textes
class CleanText:

    def __init__(self, apply_stemming=True):

        french_stopwords = nltk.corpus.stopwords.words('french')
        self.stopwords = [self.remove_accent(sw) for sw in french_stopwords]

        self.stemmer = nltk.stem.SnowballStemmer('french')

        self.apply_stemming = apply_stemming


    @staticmethod
    def remove_html_code(txt):
        txt = BeautifulSoup(txt, "html.parser", from_encoding='utf-8').get_text()
        return txt

    @staticmethod
    def convert_text_to_lower_case(txt):
        return txt.lower()

    @staticmethod
    def remove_accent(txt):
        return unicodedata.normalize('NFD', txt).encode('ascii', 'ignore').decode("utf-8")

    @staticmethod
    def remove_non_letters(txt):
        return re.sub('[^a-z_]', ' ', txt)

    def remove_stopwords(self, txt):
        return [w for w in txt.split() if (w not in self.stopwords)]

    def get_stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
    
    
## Mise en application des processus de nettoyage 
ct = CleanText()
def apply_all_transformation(txt):
    txt = ct.remove_html_code(txt)
    txt = ct.convert_text_to_lower_case(txt)
    txt = ct.remove_accent(txt)
    txt = ct.remove_non_letters(txt)
    tokens = ct.remove_stopwords(txt)
    tokens_stem = ct.get_stem(tokens)
    return tokens_stem


## Gestion des données sous la forme d'un dataframe, on ajoute une colomn qui contient le text nettoyé
def clean_df_column(df, column_name, clean_column_name):
    df[clean_column_name] = [" ".join(apply_all_transformation(x)) for x in tqdm(df[column_name].values)]
        
        