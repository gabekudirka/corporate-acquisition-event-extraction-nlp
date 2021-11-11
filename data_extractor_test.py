import os
import spacy
import json
from spacy.tokens.doc import Doc

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

    def process_doc(self):
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(self.doc)
        self.processed_doc = doc

        valid_entities = []
        for entity in doc.ents:
            if 'and' not in entity.text:
                entity.label_ = entity.label_.strip()
                valid_entities.append(entity)
            
        self.entities = valid_entities
        
    def read_doc(self, filepath):
        text_raw = open(filepath, "r").read()
        return text_raw.replace('\n', ' ').strip();

    def get_acquired(self):
        for entity in self.entities:
            if entity.label_ == 'ORG' or entity.label_ == 'FACILITY':
                self.used_entities.append(entity.text)
                return '\"' + entity.text + '\"'
        return '---'

    def get_acqloc(self):
        for entity in self.entities:
            if entity.label_ == 'GPE' or entity.label_ == 'LOC':
                return '\"' + entity.text + '\"'
        return '---'

    def get_drlamt(self):
        for entity in self.entities:
            if entity.label_ == 'MONEY':
                return '\"' + entity.text + '\"'
        return '---'

    def get_purchaser(self):
        for entity in self.entities:
            if entity.label_ == 'ORG' or entity.label_ == 'PERSON':
                if entity.text not in self.used_entities:
                    return '\"' + entity.text + '\"'
        return '---'

    def get_seller(self):
        for entity in self.entities:
            if entity.label_ == 'ORG' or entity.label_ == 'PERSON':
                if entity.text not in self.used_entities:
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