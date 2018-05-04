import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim.models import hdpmodel, ldamodel
from gensim.models import CoherenceModel


from gensim import  models
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

texts = []
import csv
with open('C:/Capstone/stacksample/Tags1.csv', encoding ="utf-8") as f:
        spamreader = csv.reader(f, delimiter=' ' )
        list1 = list(spamreader)
        str2 = ' '.join(map(str, list1))
        #print(str2)

str3 = str2.lower();
print(str3)

texts = []
tokens = tokenizer.tokenize(str3)
print(tokens)


remove_numbers = [i for i in tokens if not i.isdigit()]
print(remove_numbers)

stopped_tokens = [i for i in remove_numbers if not i in en_stop]
print("----Stop words removal----")
print(stopped_tokens)

stopwords = {'got','help','set','wont','2017b','64bit','8000c','can','t','use','ubuntu','install','n','p'}
final_words = [i for i in stopped_tokens if not i in stopwords]
print("---------------Extra Removal Word------------")
print("final words",final_words)

#extra_removal_of_words = removal_of_words.split('\n')
print("Length of original list: {0} words\n"
     "Length of list after stopwords removal: {1} words"
     .format(len(tokens), len(final_words)))

#sTEMMING
stemmed_tokens = [p_stemmer.stem(i) for i in final_words]
print(stemmed_tokens)

texts.append(stemmed_tokens)

dictionary = corpora.Dictionary(texts)
#print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
#print(corpus)


#ldamodels = ldamodel.LdaModel(corpus, num_topics=15, id2word = dictionary, passes=20)
#print(ldamodels.print_topics(num_topics=15, num_words=3))

# View
#print(corpus[:1])

#print(dictionary[11])


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=200,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

print(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

#coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)
