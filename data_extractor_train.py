import os
import spacy
from spacy.tokens.doc import Doc
from data_extractor_test import Entity

class TrainDocument:
    def __init__ (self, doc_filepath, ans_filepath, get_ml_features=False):
        self.doc = self.read_doc(doc_filepath)
        self.read_ans(ans_filepath)

        if get_ml_features:
            self.get_ml_features()

    def read_doc(self, filepath):
        text_raw = open(filepath, "r").read()
        return text_raw.replace('\n', ' ').strip();

    def read_ans(self, filepath):
        ans_raw = open(filepath, "r").readlines()
        self.text = ans_raw[0][6:].strip()
        
        self.true_slots = {}
        self.acquired = ans_raw[1][10:].strip().strip('"')
        self.true_slots[self.acquired] = 'acquired'

        self.acqbus = ans_raw[2][8:].strip().strip('"')
        self.true_slots[self.acqbus] = 'acqbus'

        self.acqloc = ans_raw[3][8:].strip().strip('"')
        self.true_slots[self.acqloc] = 'acqloc'

        self.drlamt = ans_raw[4][8:].strip().strip('"')
        self.true_slots[self.drlamt] = 'drlamt'

        self.purchaser = ans_raw[5][11:].strip().strip('"')
        self.true_slots[self.purchaser] = 'purchaser'

        self.seller = ans_raw[6][8:].strip().strip('"')
        self.true_slots[self.seller] = 'seller'

        self.status = ans_raw[7][8:].strip().strip('"')
        self.true_slots[self.status] = 'status'


    def get_status(self):
        return self.status

    def get_ml_features(self):
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(self.doc)
        self.ml_features = []

        sentences = [sentence for sentence in doc.sents]
        chunks = []
        for i in range(3):
            for chunk in nlp(sentences[i].text).noun_chunks:
                chunks.append(chunk)
            
        entities = self.process_entities(doc.ents)

        for i, chunk in enumerate(chunks):
            #add base chunk features
            ml_feature = [chunk.text, chunk.root.dep_, chunk.root.head.text]
            #add second head up feature
            ml_feature.append(self.get_next_verb(chunk))
            #add position feature
            ml_feature.append(i)
            #add pos feature
            ml_feature.append(chunk.root.pos_)
            #add entity label feature
            if chunk.text in entities.keys() and entities[chunk.text].is_reference == False:
                ml_feature.append(entities[chunk.text].label)
            else:
                ml_feature.append('nonent')
            #add frequency feature
            ml_feature.append(self.doc.count(chunk.text))
            #add label
            if chunk.text in self.true_slots.keys():
                ml_feature.append(self.true_slots[chunk.text])
            else:
                ml_feature.append('none')

            self.ml_features.append(ml_feature)

    #Eventually make this recursive - use @cache
    def get_next_verb(self, chunk):
        for child in chunk.root.head.children:
            if child.pos_ == 'VERB':
                return child.text
        for child in chunk.root.head.children:
            if len(child.text) > 0 and child.text != '    ' and child.text != chunk.root.text:
                return child.text
        return chunk.root.head.text

    def process_entities(self, entities):
        valid_entities = {entity.text: Entity(entity) for entity in entities}

        for outer_entity in valid_entities.keys():
            for inner_entity in valid_entities.keys():
                if outer_entity == inner_entity or valid_entities[outer_entity].is_reference:
                    continue
                #Check if entities refer to one another
                if inner_entity in outer_entity:
                    valid_entities[outer_entity].referred_by = inner_entity
                    valid_entities[inner_entity].refers_to = outer_entity
                    valid_entities[inner_entity].is_reference = True

        return valid_entities

if __name__ == '__main__':
    test_doc = TrainDocument("./data/docs/447", "./data/anskeys/447.key", get_ml_features=True)
    print(test_doc.get_ml_features())
