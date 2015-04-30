import numpy as np
import codecs
import nltk

class RawDocs():
    
    docid = 0
    """ The RawDocs class rpresents a class of document collections
     
    """
    def __init__(self, doc_data, stopword_file):

        self.docs = []
        for doc in doc_data :
            self.docs.append(Doc(RawDocs.docid, doc[2], doc[1], doc[0]))
            RawDocs.docid += 1

        with codecs.open(stopword_file,'r','utf-8') as f: raw = f.read()
        self.stopwords = set(raw.splitlines())

        self.N = len(self.docs)
        RawDocs.docid = 0
        
    def clean_docs(self, length):
        """ 
        Applies stopword removal, token cleaning to docs
        """
        for doc in self.docs:
            doc.token_clean(length)
            doc.stopword_remove(self.stopwords)
            
    def count(self, dictionary):
        """ 
        word count frequency of dictionary in document collection
        """
        
        return ({(doc.docid, doc.year, doc.author) : \
                 doc.tf(dictionary) for doc in self.docs})
    
    def idf(self, dictionary):
        """ 
        returns array of inverted document frequency for given dictionary 
        over collection of docs
        """
        
        is_word_docs = np.array([doc.word_exists(dictionary) for doc in self.docs])
        
        return(np.log(self.N / sum([is_word for is_word in is_word_docs])))
    
    def tf_idf(self, dictionary):
        """ 
        returns tf-idf score of given dictionary of words for every document 
        """
        #tf and idf are calls to functions of the class Doc to calculate word frequency and inverse df respectively
        tf = self.count(dictionary)
        idf = self.idf(dictionary)
        
        tf_idf_docs = dict()
        
        for doc in self.docs:
            tf_idf_docs[(doc.docid, doc.year, doc.author) ] = \
            np.log(tf[(doc.docid, doc.year, doc.author)] + 1) * idf
            
        return(tf_idf_docs)
    
    def rank_tfidf(self, dictionary):
        
        """
        Calculates document rank based on tfidf
        """
        
        docs_tfidf = self.tf_idf(dictionary)
        
        doc_rank = [[key, sum(docs_tfidf[key])] for key in docs_tfidf.keys()]
        
        doc_rank.sort(key=lambda x: x[1], reverse = True)
        
        return(doc_rank)   
        #return(np.sort(np.array(doc_rank), axis=0)[::-1])
    
    def rank_count(self, dictionary):
        
        """
        Calculates document rank based on word frequency
        """
        
        docs_count = self.count(dictionary)
        
        doc_rank = [[key, sum(docs_count[key])] for key in docs_count.keys()]
        
        doc_rank.sort(key=lambda x: x[1], reverse = True)
        
        return(doc_rank)
        #return(np.sort(np.array(doc_rank), axis=0)[::-1])



import numpy as np
import re
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer

class Doc():
    
    """ The Doc class rpresents a class of individula documents
    
    """
    
    def __init__(self, docid, doc, author, year):
        self.docid = docid
        self.text = doc.lower()
        self.text = re.sub(u'[\u2019\']', '', self.text)
        self.tokens = np.array(wordpunct_tokenize(self.text))
        self.stem = None
        self.author = author
        self.year = year
        
    def tf(self, wordlist):
        
        """
        Returns ARRAY with wordlist frequency
        """
        
        count = np.zeros(len(wordlist))
        
        for wid, word in np.ndenumerate(wordlist):
            count[wid] = (self.tokens == word).sum()
        return count
        
    
    def word_exists(self, wordlist):
        
        """
        Returns ARRAY of binary value where 1 inidicates presence of a word
        """
        
        is_word = np.zeros(len(wordlist))
        
        for wid, word in np.ndenumerate(wordlist):
            if word in self.tokens:
                is_word[wid] = 1
        return is_word
            
    def token_clean(self,length):

        """ 
        strip out non-alpha tokens and length one tokens
        """

        self.tokens = np.array([t for t in self.tokens if (t.isalpha() and len(t) > length)])


    def stopword_remove(self, stopwords):

        """
        Remove stopwords from tokens.
        """

        
        self.tokens = np.array([t for t in self.tokens if t not in stopwords])


    def stem(self):

        """
        Stem tokens with Porter Stemmer.
        """
        
        self.stems = n.array([PorterStemmer().stem(t) for t in self.tokens])


