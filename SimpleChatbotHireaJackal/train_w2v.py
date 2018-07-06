import os,sys
import codecs
from gensim.models import Phrases
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import re
import multiprocessing
from gensim.models.word2vec import Word2Vec


#CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(CURRENT_DIR))



class WikiMedicalIter(object):

    no_of_Sentence = 0
    All_text=list()
    filteredText=list()
    def __init__(self, directory, sentence_tokenize=False):
        self.directory = directory
        self.sentence_tokenize = sentence_tokenize

    def __iter__(self):
        print('pass...')
        # traverse root directory, and list directories as dirs and files as files
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                file_name = os.path.join(root, file)
                file_present = os.path.isfile(file_name)
                filename, file_extension = os.path.splitext(file)

                if (file_extension == '.txt' or file_extension == '.csv') and file_present:
                    # print filename
                    file_wiki = codecs.open(file_name, "r", "utf-8")
                    content = file_wiki.read()
                    file_wiki.close()

                    if self.sentence_tokenize:
                        sent_tokenize_list = sent_tokenize(content)
                        for sent in sent_tokenize_list:
                            words = self.sentenceCleaner(sent)
                            if len(words) > 2:
                                yield words
                    else:
                        yield content.lower().split()

    def sentenceCleaner(self, sentence, remove_stopwords=True, clean_special_chars_numbers=True, \
                        lemmatize=True, stem=False, stops=set(stopwords.words("english"))):
        """
        Function to convert a document to a sequence of words, optionally removing stop words.  Returns a list of words.

        :param sentence:
        :param remove_stopwords:
        :param clean_special_chars_numbers:
        :param lemmatize:
        :param stem:
        :param stops:
        :return:
        """
        words = []

        if sentence.startswith('==') == False:
            sentence_text = sentence
            self.All_text.append(sentence_text)
            self.no_of_Sentence=self.no_of_Sentence+1
            # Optionally remove non-letters (true by default)
            if clean_special_chars_numbers:
                sentence_text = re.sub("[^a-zA-Z]", " ", sentence_text)

            # Convert words to lower case and split them
            words = sentence_text.lower().split()

            # Optional stemmer
            if stem:
                stemmer = PorterStemmer()
                words = [stemmer.stem(w) for w in words]

            if lemmatize:
                lemmatizer = WordNetLemmatizer()
                words = [lemmatizer.lemmatize(w) for w in words]

            # Optionally remove stop words (false by default)
            if remove_stopwords:
                words = [w for w in words if not w in stops]

            self.filteredText.append(words)

        # 4. Return a list of words
        return (words)

    def fileCounter(self):
        file_counter = 0
        for root, dirs, files in os.walk(self.directory):
            file_counter += len(files)
        return file_counter



folder_name = "Corpus"
print(folder_name)
#print CURRENT_DIR

iterator=WikiMedicalIter(folder_name,True)
noOfArticles = iterator.fileCounter()

item=Phrases(iterator)

print('Given folder has', noOfArticles, 'articles.') #263

print("No. of Sentences:",iterator.no_of_Sentence) #35263

bigram_transformer = Phrases(iterator)
trigram_transformer = Phrases(bigram_transformer[iterator])

# Set values for various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 10  # Minimum word count
num_workers = multiprocessing.cpu_count()  # Number of threads to run in parallel
context = 10  # Context window size
downsampling =  1e-3  # Downsample setting for frequent words

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
SoftModel_w2v = Word2Vec(size=300, min_count=10)

SoftModel_w2v.build_vocab([x.split() for x in tqdm(iterator.All_text)])
SoftModel_w2v.train([x.split() for x in tqdm(iterator.All_text)],total_examples=iterator.All_text.__len__(),word_count=5,epochs=150)

SoftModel_w2v1 = Word2Vec(size=300, min_count=10)
SoftModel_w2v1.build_vocab([x for x in tqdm(iterator.filteredText)])
SoftModel_w2v1.train([x for x in tqdm(iterator.filteredText)],total_examples=iterator.filteredText.__len__(),word_count=5,epochs=150)
