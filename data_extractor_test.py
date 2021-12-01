import os
import spacy
import json
from spacy.tokens import Span
from spacy.tokens.doc import Doc
from spacy.matcher import PhraseMatcher

class Entity:
    def __init__(self, entity):
        self.name = entity.text
        self.label = entity.label_
        self.start = entity.start_char
        self.end = entity.end_char
        self.refers_to = None
        self.referred_by = None
        self.is_reference = False

class Entity_:
    def __init__(self, text, label, start, end):
        self.text = text
        self.label = label
        self.start = start
        self.end = end
        self.refers_to = None
        self.referred_by = None
        self.is_reference = False


class TestDocument:
    def __init__ (self, doc_filepath):
        self.doc = self.read_doc(doc_filepath)
        self.process_doc()

        self.used_entities = []
        self.text = os.path.basename(doc_filepath)
        self.acquired = self.get_acquired()
        self.acqbus = '---'
        self.acqloc = self.get_acqloc()
        self.drlamt = self.get_drlamt()
        self.purchaser = self.get_purchaser()
        self.seller = self.get_seller()
        self.status = self.set_status(doc_filepath)

    def read_doc(self, filepath):
        text_raw = open(filepath, "r").read()
        return text_raw.replace('\n', ' ').strip()

    def process_doc(self):
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(self.doc)
        self.processed_doc = doc

        #Add custom entities
        dlramt_phrase_list = ['undisclosed', 'not disclosed']
        doc.ents = self.add_custom_entity(nlp, doc, dlramt_phrase_list, 'UNDSCLSD')

        self.process_entities(doc.ents)
        self.sentences = [sentence for sentence in doc.sents]

        sentence = nlp(self.sentences[0].text)
        chunks = [(chunk.text, chunk.root.text, chunk.root.dep_,
            chunk.root.head.text) for chunk in sentence.noun_chunks]

        self.chunks = sentence.noun_chunks
        self.chunks_text = [chunk.text for chunk in self.chunks]

    #Format and set entities
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

        #set entity dicts for each type
        self.loc_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'LOC' or entity[1].label == 'GPE'}
        self.money_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'MONEY'}
        self.acquired_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'ORG' or entity[1].label == 'FACILITY'}
        self.buyer_seller_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'ORG' or entity[1].label == 'PERSON'}
        self.entities = valid_entities

    def add_custom_entity(self, nlp, doc, phrase_list, label):
        matcher = PhraseMatcher(nlp.vocab)
        phrase_patterns = [nlp(text) for text in phrase_list]
        matcher.add(label, None, *phrase_patterns)
        matches = matcher(doc)

        if label == "UNDSCLSD":
            lbl = doc.vocab.strings[u'UNDSCLSD']

        new_ents = [Span(doc, match[1],match[2],label= lbl) for match in matches]
        doc.ents = list(doc.ents) + new_ents

        return doc.ents

    def get_acquired(self):
        for entity in self.entities:
            if entity.label_ == 'ORG' or entity.label_ == 'FACILITY':
                self.used_entities.append(entity.text)
                return '\"' + entity.text + '\"'
        return '---'

    def get_possible_acqloc(self, entities, sentence, sent_idx, chunks):
        #include noun chunks or naw?                                                                                                                                      
        #put the location concatenation code here                                                                                                                         
        used_chunks = list(entities.keys())
        lst = []
        loc_candidates = []
        for item in entities.items():
            if item[1].label == 'GPE' or item[1].label == 'LOC' or item[1].label == 'LANGUAGE' or item[1].label == 'NORP': #NORP is a maybe                               
                lst.append(item)


        concatenated_locations = []
        for i in range(1, len(lst)):
            current = lst[i]
            previous = lst[i-1]

            if i == len(lst):
                current_ = Entity_(current[1].text, current[1].label, current[1].start, current[1].end)
                concatenated_locations.append(current_)

            elif previous[1].end >= (current[1].start - 5):
                text = previous[1].text + sentence.text[previous[1].end : current[1].start] + current[1].text
                start = previous[1].start
                end = current[1].end
                concatenated_item = Entity_(text, 'LOC', start, end)
                concatenated_locations.append(concatenated_item)
                if i == len(lst)-1:
                    break
            else:
                previous_ = Entity_(previous[1].text, previous[1].label, previous[1].start, previous[1].end)
                concatenated_locations.append(previous_)

        for item in concatenated_locations:
            print(item.text)
            chunk = self.match_to_chunk(item.text, chunks, sentence)
            loc_candidates.append(FeatureExtractor(chunk, self.doc, sentence, sent_idx, entity=item))
            used_chunks.append(chunk.text)


        return loc_candidates


    def get_drlamt(self):
        for entity in self.entities:
            if entity.label_ == 'MONEY' or entity.label_ == 'UNDSCLSD':
                return '\"' + entity.text + '\"'
        return '---'

    def get_purchaser(self):
        for entity in self.entities:
            if entity.label_ == 'ORG' or entity.label_ == 'PERSON':
                #if entity.text not in self.used_entities:
                    return '\"' + entity.text + '\"'
        return '---'

    def get_seller(self):
        for entity in self.entities:
            if entity.label_ == 'ORG' or entity.label_ == 'PERSON':
                #if entity.text not in self.used_entities:
                    return '\"' + entity.text + '\"'
        return '---'

    def set_status(self, filepath):
        with open('all_statuses.json') as statuses_json:
            status_dict = json.load(statuses_json)

        with open(filepath, "r") as d:
            data = d.read()

        for status in status_dict:
            if status in data:
                return '\"' + status + '\"'

        return '---'

    def print_to_file(self, output_file):
        f = open(output_file, "a")   
        f.write('TEXT: ' + self.text + '\n')
        f.write('ACQUIRED: ' + self.acquired + '\n')
        f.write('ACQBUS: ' + self.acqbus + '\n') 
        f.write('ACQLOC: ' + self.acqloc + '\n')
        f.write('DLRAMT: ' + self.drlamt + '\n')
        f.write('PURCHASER: ' + self.purchaser + '\n')
        f.write('SELLER: ' + self.seller + '\n') 
        f.write('STATUS: ' + self.status + '\n \n')
        f.close()