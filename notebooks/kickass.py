import numpy as np
import codecs
import nltk

class ProjectCollection():
    
    """ The RawDocs class rpresents a class of document collections
     
    """
    def __init__(self, doc_data, stopword_file):

        self.docs = [Project(doc) for doc in doc_data] 

        with codecs.open(stopword_file,'r','utf-8') as f: raw = f.read()
        self.stopwords = set(raw.splitlines())

        self.N = len(self.docs)
        
    def clean_docs(self, length):
        """ 
        Applies stopword removal, token cleaning to docs
        """
        for doc in self.docs:
            doc.token_clean(length)
            doc.stopword_remove(self.stopwords)
            doc.stem()
            
    def count(self, dictionary):
        """ 
        word count frequency of dictionary in document collection
        """
        
        return ({(doc.pid) : \
                 doc.tf(dictionary) for doc in self.docs})
    
    def idf(self, dictionary):
        """ 
        returns array of inverted document frequency for given dictionary 
        over collection of docs
        """
        
        is_word_docs = np.array([doc.word_exists(dictionary) for doc in self.docs])
        
        asum = sum([is_word for is_word in is_word_docs])
        asum = sum(asum)
        if asum != 0:
            idf_list = np.log(self.N / asum )
        else:
            idf_list = np.zeros(len(dictionary))
            
        return idf_list
    
    def tf_idf(self, dictionary):
        """ 
        returns tf-idf score of given dictionary of words for every document 
        """
        #tf and idf are calls to functions of the class Doc to calculate word frequency and inverse df respectively
        tf = self.count(dictionary)
        idf = self.idf(dictionary)
        
        tf_idf_docs = dict()
        
        for doc in self.docs:
            tf_idf_docs[doc.pid] = \
            np.log(tf[doc.pid] + 1) * idf
            
        return(tf_idf_docs)
    
    def rank_tfidf(self, dictionary, top):
        
        """
        Calculates document rank based on tfidf
        """
        
        docs_tfidf = self.tf_idf(dictionary)
        
        doc_rank = [[key, sum(docs_tfidf[key])] for key in docs_tfidf.keys()]
        
        doc_rank.sort(key=lambda x: x[1], reverse = True)
        
        return(doc_rank[0:(top + 1)])   
        #return(np.sort(np.array(doc_rank), axis=0)[::-1])
    
    def rank_count(self, dictionary, top):
        
        """
        Calculates document rank based on word frequency
        """
        
        docs_count = self.count(dictionary)
        
        doc_rank = [[key, sum(docs_count[key])] for key in docs_count.keys()]
        
        doc_rank.sort(key=lambda x: x[1], reverse = True)
        
        return(doc_rank[0:(top+1)])
        #return(np.sort(np.array(doc_rank), axis=0)[::-1])



import numpy as np
import re
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer

class Project():
    
    """ The Doc class rpresents a class of individula documents
    
    """
    
    def __init__(self, project_dict):
        self.pid = project_dict['id']
        self.blurb = project_dict['blurb'].lower()
        self.deadline = project_dict['deadline']
        self.category = project_dict['category'] 
        self.reward_backer_tup = project_dict['reward_backer_tup'] 
        #self.risk = project_dict['risk'][0].lower()
        self.risk = re.sub(u'[\u2019\']', '', self.pre_process(project_dict['risk']))
        self.risk_tokens = np.array(wordpunct_tokenize(self.risk))
        self.name = project_dict['name'] 
        self.url = project_dict['url'] 
        self.launched_at = project_dict['launched_at'] 
        self.pledged = project_dict['pledged']
        self.title = project_dict['title']
        self.no_dollars_raised = project_dict['no_dollars_raised']
        self.currency = project_dict['currency']
        self.no_backers = project_dict['no_backers']
        self.state = project_dict['state']
        self.deadline = project_dict['deadline']
        self.location = project_dict['location']
        self.backers_count = project_dict['backers_count']
        self.creator_url = project_dict['creator_url']
        self.backers_count = project_dict['backers_count']
        self.spotlight = project_dict['spotlight']
        self.goal = project_dict['goal']
        self.author = project_dict['author']
        self.stems = None
        
        
    def pre_process(self, list_text):
        sentence = ""
        for sent in list_text:
            sentence += " " + sent.lower()
        return sentence
            
    def tf(self, wordlist):
        
        """
        Returns ARRAY with wordlist frequency
        """
        
        count = np.zeros(len(wordlist))
        
        for wid, word in np.ndenumerate(wordlist):
            count[wid] = (self.risk_tokens == word).sum()
        return count
        
    
    def word_exists(self, wordlist):
        
        """
        Returns ARRAY of binary value where 1 inidicates presence of a word
        """
        
        is_word = np.zeros(len(wordlist))
        
        for wid, word in np.ndenumerate(wordlist):
            if word in self.risk_tokens:
                is_word[wid] = 1
        return is_word
            
    def token_clean(self,length):

        """ 
        strip out non-alpha tokens and length one tokens
        """

        self.risk_tokens = np.array([t for t in self.risk_tokens if (t.isalpha() and len(t) > length)])


    def stopword_remove(self, stopwords):

        """
        Remove stopwords from tokens.
        """

        
        self.risk_tokens = np.array([t for t in self.risk_tokens if t not in stopwords])


    def stem(self):

        """
        Stem tokens with Porter Stemmer.
        """
        
        self.stems = np.array([PorterStemmer().stem(t) for t in self.risk_tokens])


