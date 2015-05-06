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
        
        #stopword removal, token cleaning and stemming to docs
        self.clean_docs(2)
        
        #creates a set of all doc tokens
        self.docs_tokens = self.create_docs_tokens()
        
    def clean_docs(self, length):
        """ 
        Applies stopword removal, token cleaning and stemming to docs
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
        asum_sum = sum(asum)
        
        if asum_sum != 0:
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

        
    def create_docs_tokens(self):
        """
        Creates a set of tokens from all documents
        """
        
        docs_tokens = set()
        
        for doc in self.docs:
            docs_tokens.update(doc.risk_tokens)
            
        return docs_tokens
    
    def doc_term_mat(self):
        """
        Reurns a list of doccument ids and a document term matrix weighted by the tfidf weighting
        """
        #applies tf-idf weighting to docs_tokens
        docs_list_tfidf = self.tf_idf(list(self.docs_tokens))
        
        #create lists of docids and their tfidf weighting with respect to docs_tokens
        docs_id = []
        docs_term_mat = []
        for key, value in docs_list_tfidf.items():
            docs_id.append(key)
            docs_term_mat.append(value)
            
        return docs_id, docs_term_mat
    
    def svd(self, cutoff):
        """
        The svd function performs a SVD analysis of the document term matrix of the collection of docs.
        It takes a parameter cutoff which behaves the following way:
        1] if cutoff == 1, returns the U,s,V matrices of the entire document term matrix
        2] if 0 <= cutoff < 1, returns a lower dimension approximation of U, s, V that corresponds to 
           cutoff % of the singular values
        3] if cutoff > 1, returns a lower dimension approximation of U, s, V that corresponds to the largest
           cutoff singular values
        
        """
        if cutoff > self.N:
            return "Error, cutoff is greater than singular values"
        
        else:

            docs_id, docs_term_mat = self.doc_term_mat()

            #apply svd to the doc - term matrix 
            D, s, T = np.linalg.svd(np.array(docs_term_mat))

            #apply lower dimension rank approximation based on parameter cutoff
            #if cut off  k between 0 and 1, then return ksingular values and singular vectors
            #else return the first k singular valuesa nd vectors

            if cutoff != 1:
                #calculate variance explained
                var = np.cumsum(s ** 2) / sum(s ** 2)

                if (cutoff >= 0 and cutoff< 1):

                    index = np.argmax(var > cutoff)
                    D = D[:, 0:index]
                    s = s[0:index]
                    T = T[0:index, :]
                elif cutoff > 1:

                    D = D[:, 0:cutoff]
                    s = s[0:cutoff]
                    T = T[0:cutoff, :]

            return (docs_id, D, s, T)

    
    def latent_doc_term_mat(self, cutoff):
        """
        Returns a doc list and a lower dimesnion approximation of the document term value based on the cutoff
        1] if cutoff == 1, returns the original document term  matrix
        2] if 0 <= cutoff < 1, returns a lower dimension approximation of the document term  matrix that corresponds to 
           cutoff % of the singular values
        3] if cutoff > 1, returns a lower dimension approximation of the document term  matrix that corresopnds to the largest
           cutoff singular values
        
        """
        docs_id, D, s, T = self.svd(cutoff)
        
        S = np.diag(s)
        
        return docs_id, np.dot(D, np.dot(S, T))
    
    
    def norm_matrix(self, mat):
        """
        Normalises a matrix of row vectors
        
        NOTE TO SELF: extend method to normalise single vectors
        """
        norm = np.sqrt(np.square(mat).sum(axis = 1))
        return  (mat / norm[:,None])
    
    def cosine_sim(self, dt_mat):
        """
        calculates cosine similarties of documents in a doc term matrix
        
        NOTE TO SELF: extend mehod to handle similarities between in sample docs and a out of sample doc
        """
        norm = self.norm_matrix(dt_mat)
        return np.dot(norm, norm.T)
        
        


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
        
        self.risk_tokens = np.array([PorterStemmer().stem(t) for t in self.risk_tokens])


