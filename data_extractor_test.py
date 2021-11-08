import os
import spacy
from spacy.tokens.doc import Doc
from set_status import set_status

class TestDocument:
    def __init__ (self, doc_filepath):
        self.filepath = doc_filepath
        self.doc = self.read_doc(doc_filepath)
        self.entities = self.extract_entities()

        self.text = self.get_text(doc_filepath)
        self.acquired = self.get_acquired()
        self.acqbus = '---'
        self.acqloc = self.get_acqloc()
        self.drlamt = self.get_drlamt()
        self.purchaser = self.get_purchaser()
        self.seller = self.get_seller()
        #self.status = self.get_status()                                              
        self.status = '---'

    def extract_entities(self):
        nlp = spacy.load("en_core_web_trf")
        text_model = nlp(self.doc)
        entities = []
        valid_entities = []

        for entity in text_model.ents:
            entities.append(entity)

        for entity in entities:
            if 'and' not in entity.text:
                entity.label_ = entity.label_.strip()
                valid_entities.append(entity)

        return valid_entities

    def read_doc(self, filepath):
        text_raw = open(filepath, "r").read()
        return text_raw.replace('\n', ' ').strip();

    def get_filepath(self):
        return self.filepath

    def get_text(self, filepath):
        file = os.path.basename(filepath)
        return os.path.splitext(file)[0]

    def get_acquired(self):
        for entity in self.entities:
            if entity.label_ == 'ORG' or entity.label_ == 'FACILITY':
                return entity.text
        return '---'

    def get_acqloc(self):
        for entity in self.entities:
            if entity.label_ == 'GPE' or entity.label_ == 'LOC':
                return entity.text
        return '---'

    def get_drlamt(self):
        for entity in self.entities:
            if entity.label_ == 'MONEY':
                return entity.text
        return '---'
        
    def get_purchaser(self):
        for entity in self.entities:
            if entity.label_ == 'ORG' or entity.label_ == 'PERSON':
                return entity.text
        return '---'

    def get_seller(self):
        for entity in self.entities:
            if entity.label_ == 'ORG' or entity.label_ == 'PERSON':
                return entity.text
        return '---'

    def get_status(self):

        return self.status

    def set_status(self):
        self.status = set_status(self.filepath)