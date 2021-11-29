import os
import json
import spacy
from spacy.tokens.doc import Doc
import string
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import csv
import gensim.downloader
import numpy as np
import re

class Entity:
    def __init__(self, entity):
        self.name = entity.text
        self.label = entity.label_
        self.start = entity.start_char
        self.end = entity.end_char
        self.refers_to = None
        self.referred_by = None
        self.is_reference = False

    def create_entity_feature_vec(self):
        pass

spacy_model = spacy.load('en_core_web_trf')
gensim_model = gensim.downloader.load('glove-wiki-gigaword-50')

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
        return text_raw.replace('\n', ' ').strip();

    def process_doc(self):
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(self.doc)
        self.processed_doc = doc

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
        self.loc_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'LOC' or entity[1].label == 'GPE' or entity[1].label == 'LANGUAGE'}
        self.money_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'MONEY'}
        self.acquired_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'ORG' or entity[1].label == 'FACILITY'}
        self.buyer_seller_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'ORG' or entity[1].label == 'PERSON'}
        self.entities = valid_entities

    def process_entities_new(self, entities):
        valid_entities = []


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


if __name__ == '__main__':
    test_doc = TestDocument("./data/docs/2434")
    features = test_doc.get_feature_vector()
    print(features)