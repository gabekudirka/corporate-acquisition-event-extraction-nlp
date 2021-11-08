import os
import spacy
from spacy.tokens.doc import Doc

class TrainDocument:
    def __init__ (self, doc_filepath, ans_filepath):
        self.doc = self.read_doc(doc_filepath)
        self.read_ans(ans_filepath)
        
    def read_doc(self, filepath):
        text_raw = open(filepath, "r").read()
        return text_raw.replace('\n', ' ').strip();

    def read_ans(self, filepath):
        ans_raw = open(filepath, "r").readlines()
        self.text = ans_raw[0][6:].strip()
        self.acquired = ans_raw[1][10:].strip('"').strip()
        self.acqbus = ans_raw[2][8:].strip('"').strip()
        self.acqloc = ans_raw[3][8:].strip('"').strip()
        self.drlamt = ans_raw[4][8:].strip('"').strip()
        self.purchaser = ans_raw[5][11:].strip('"').strip()
        self.seller = ans_raw[6][8:].strip('"').strip()
        self.status = ans_raw[7][8:].strip('"').strip()

class TestDocument:
    def __init__ (self, doc_filepath):
        self.doc = self.read_doc(doc_filepath)
        self.process_doc()

        self.text = os.path.basename(doc_filepath)
        self.acquired = self.get_acquired()
        self.acqbus = '---'
        self.acqloc = self.get_acqloc()
        self.drlamt = self.get_drlamt()
        self.purchaser = self.get_purchaser()
        self.seller = self.get_seller()
        self.status = '---'

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
