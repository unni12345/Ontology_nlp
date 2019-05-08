import spacy
import random


TRAIN_DATA = [('The Three Phase transformer should be of variable output (autotransformer-like) isolation type.', {'entities': [ (4 , 27 , 'machine')  ]}), ('The secondary shall be wound over the primary with suitable insulation between them.', {'entities': [(4 , 13 , 'component') ]}), ('The secondary output shall be tapped by means of a brush arm moving on it.', {'entities': [(4 , 20 , 'component') ]}), ('Brushes for the three phase windings shall be made to move in tandem by means of a common arm.', {'entities': [(0 , 7 , 'component') ]}), ('The general overall size shall not exceed 600 mm (height) X 300 mm (width) X 300 mm (depth).', {'entities': [(42 , 91 , 'dimension')]}), ('The mechanical arrangement shall be such that the rotating shaft (at the user end) horizontal.', {'entities': [(50 , 64 , 'component')]})]


def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(TRAIN_DATA, 20)

# Save our trained Model
count = 0
while (count < 1):
    count = count + 1
    modelfile = input("Enter your Entity identifier/name ")

    prdnlp.to_disk(modelfile)

    #Test your text
    test_text = input("Enter your testing text: ")
    doc = prdnlp(test_text)
    for ent in doc.ents:
         print(ent.text, ent.start_char, ent.end_char, ent.label_)
