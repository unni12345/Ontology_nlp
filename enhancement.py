import spacy

from spacy_wordnet.wordnet_annotator import WordnetAnnotator 
# Load an spacy model (supported models are "es" and "en") 
nlp = spacy.load('en')
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
token = nlp('prices')[0]

# wordnet object link spacy token with nltk wordnet interface by giving acces to
# synsets and lemmas 
token._.wordnet.synsets()
token._.wordnet.lemmas()

# And automatically tags with wordnet domains
token._.wordnet.wordnet_domains()

# Imagine we want to enrich the following sentence with synonyms
sentence = nlp('The Three Phase transformer should be of variable output (autotransformer-like) isolation type. The secondary shall be wound over the primary with suitable insulation between them. The secondary output shall be tapped by means of a brush arm moving on it. Brushes for the three phase windings shall be made to move in tandem by means of a common arm. The general overall size shall not exceed 600 mm (height) X 300 mm (width) X 300 mm (depth). The mechanical arrangement shall be such that the rotating shaft (at the user end) horizontal.')


# spaCy WordNet lets you find synonyms by domain of interest
# for example economy
engineering_domains = ['engineering', 'technology']
enriched_sentence = []

# For each token in the sentence
for token in sentence:
    # We get those synsets within the desired domains
    synsets = token._.wordnet.wordnet_synsets_for_domain(engineering_domains)
    print( token, "\t", token.pos_)
    if synsets:
        lemmas_for_synset = []
        for s in synsets:
            # If we found a synset in the economy domains
            # we get the variants and add them to the enriched sentence
            
            lemmas_for_synset.extend(s.lemma_names())
            print(lemmas_for_synset)
            print("\n")            
            enriched_sentence.append('({})'.format('|'.join(set(lemmas_for_synset))))
    else:
        enriched_sentence.append(token.text)
        

# Let's see our enriched paragraph
print(' '.join(enriched_sentence))
