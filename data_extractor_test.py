import os
import json
import spacy
from spacy.matcher import Matcher
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import csv
import gensim.downloader
import numpy as np
import string

class Entity:
    def __init__(self, entity):
        self.entity = entity
        self.text = entity.text
        self.label = entity.label_
        self.start = entity.start_char
        self.end = entity.end_char
        self.refers_to = None
        self.referred_by = None
        self.is_reference = False

    def create_entity_feature_vec(self):
        pass


def get_location_sentence(sentence, phrase):
        try:
            start_index = sentence.text.index(phrase)
            start_loc = len(sentence.text[:start_index].split())
            return (start_loc, start_loc + len(phrase.split()))
        except:
            return (-1, -1)

gensim_model = gensim.downloader.load('glove-wiki-gigaword-100')

class FeatureExtractor():
    def __init__(self, candidate, doc, sentence, sent_idx, entity=None):
        #Make sure to only use non reference entities
        self.gensim_model = gensim_model
        if hasattr(candidate, 'root'):
            self.candidate = candidate.root
        else:
            self.candidate = candidate

        if entity is not None:
            self.entity_label = entity.label
            self.text = entity.text
        else:
            self.entity_label = 'nonent'
            self.text = candidate.text

        self.dep = self.candidate.dep_
        self.head_pos = self.candidate.head.pos_
        self.head_text = self.candidate.head.text
        self.head_dep = self.candidate.head.dep_
        self.freq_in_doc = doc.count(self.text)
        self.sentence_loc, _ = get_location_sentence(sentence, self.text)
        self.sentence_num = sent_idx
        self.next_verb = self.get_next_verb(self.candidate, sentence)
        self.sliding_window = self.get_sliding_window(self.text, sentence)

        self.text_embedding = self.get_word_embeddings(self.text)
        self.head_text_embedding = self.get_word_embeddings(self.head_text)
        self.next_verb_embedding = self.get_word_embeddings(self.next_verb)
        self.sliding_window_embeddings = [self.get_word_embeddings(word) for word in self.sliding_window]
        self.one_hot_encode()

    def get_feature_vector(self):
        feature_vec_output = [self.freq_in_doc, self.sentence_loc, self.sentence_num, self.dep_one_hot, self.head_pos_one_hot,
                              self.head_dep_one_hot, self.ent_label_one_hot, self.text_embedding, self.head_text_embedding, self.next_verb_embedding]
        feature_vec_output.extend(self.sliding_window_embeddings)

        return feature_vec_output

    def one_hot_encode(self):
        possible_pos_arr = ['PROPN', 'NUM', 'SPACE', 'ADJ', 'PART', 'PUNCT', 'AUX', 'PRON', 'VERB', 'CCONJ', 'ADP', 'ADV', 'NOUN', 'DET', 'SCONJ']
        possible_dep_arr = ['amod', 'relcl', 'dobj', 'agent', 'appos', 'prep', 'dep', 'cc', 'attr', 'acl', 'auxpass', 'compound', 'advmod', 'ROOT', 'nmod', 'case', 'conj', 'ccomp', 'poss', 'nsubj', 'det', 'pcomp', 'npadvmod', 'aux', 'xcomp', 'oprd', 'parataxis', 'nsubjpass', 'punct', 'dative', 'quantmod', 'acomp', 'nummod', 'pobj', 'neg', 'advcl']
        possible_ent_labels_arr = ['nonent', 'ORG', 'GPE', 'PERSON', 'FAC', 'WORK_OF_ART', 'LANGUAGE', 'MONEY']
        possible_pos = {}
        possible_dep = {}
        possible_ent_labels = {}
        for i, pos in enumerate(possible_pos_arr):
            possible_pos[pos] = i
        for i, dep in enumerate(possible_dep_arr):
            possible_dep[dep] = i
        for i, ent_label in enumerate(possible_ent_labels_arr):
            possible_ent_labels[ent_label] = i
        
        self.dep_one_hot = np.zeros(len(possible_dep_arr)).tolist()
        self.head_pos_one_hot = np.zeros(len(possible_pos_arr)).tolist()
        self.head_dep_one_hot = np.zeros(len(possible_dep_arr)).tolist()
        self.ent_label_one_hot = np.zeros(len(possible_ent_labels_arr)).tolist()

        if self.dep in possible_dep:
            self.dep_one_hot[possible_dep[self.dep]] = 1
        if self.head_pos in possible_pos:
            self.head_pos_one_hot[possible_pos[self.head_pos]] = 1
        if self.head_dep in possible_dep:
            self.head_dep_one_hot[possible_dep[self.head_dep]] = 1
        if self.entity_label in possible_ent_labels:
            self.ent_label_one_hot[possible_ent_labels[self.entity_label]] = 1

    def get_word_embeddings(self, word):
        if len(word.split()) == 1:
            word = word.translate(str.maketrans('', '', string.punctuation))
            if word.isdigit():
                return self.gensim_model['number']
            try:
                return self.gensim_model[word]
            except:
                return self.gensim_model['unknown']
        elif len(word.split()) > 1:
            word_tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in word.split()]
            embeddings = []
            for word in word_tokens:
                if word.isdigit():
                    embeddings.append(self.gensim_model['number'])
                    continue
                try:
                    embeddings.append(self.gensim_model[word])
                except:
                    embeddings.append(self.gensim_model['unknown'])
            return np.mean(np.asarray(embeddings), axis=0)
        else:
            return self.gensim_model['unknown']

    #get two word sliding window
    def get_sliding_window(self, phrase, sentence):
        sentence_tokens = sentence.text.split()
        indexes = get_location_sentence(sentence, phrase)

        sliding_window = []
        l = len(sentence_tokens)
        try:
            if indexes[0] == 0 or indexes[0] == -1:
                sliding_window.append('phi2')
                sliding_window.append('phi1')
            elif indexes[0] == 1:
                sliding_window.append('phi1')
                sliding_window.append(sentence_tokens[0])
            else:
                sliding_window.append(sentence_tokens[indexes[0]-2])
                sliding_window.append(sentence_tokens[indexes[0]-1])
        except:
            sliding_window = ['phi2', 'phi1']

        try:
            if indexes[1] == len(sentence_tokens) or indexes[1] == -1:
                sliding_window.append('omega1')
                sliding_window.append('omega2')
            elif indexes[1] + 1 == len(sentence_tokens):
                sliding_window.append(sentence_tokens[indexes[1]])
                sliding_window.append('omega1')
            else:
                sliding_window.append(sentence_tokens[indexes[1]])
                sliding_window.append(sentence_tokens[indexes[1]+1])
        except:
            sliding_window = sliding_window[:2]
            sliding_window.append('omega1')
            sliding_window.append('omega2')

        return sliding_window

    def get_next_verb(self, target, sentence):
        if hasattr(target, 'root'):
            target = target.root

        edges = [] 
        for token in sentence:
            for child in token.children:
                edges.append((token, child))
        graph = nx.Graph(edges)

        next_verb = token.head
        verbs = [token for token in sentence if token.pos_ == 'VERB']
        shortest_path_length = 99999
        for verb in verbs:
            if verb == target:
                continue
            try:
                length = nx.shortest_path_length(graph, target, verb)
            except nx.NetworkXNoPath:
                continue
            if length < shortest_path_length:
                shortest_path_length = length
                next_verb = verb
        return next_verb.text
                
spacy_model = spacy.load('en_core_web_trf')

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

    #Format and set entities
    def process_entities(self, doc):
        entities = doc.ents

        valid_entities = {entity.text: Entity(entity) for entity in entities}

        for outer_entity in valid_entities.keys():
            for inner_entity in valid_entities.keys():
                if outer_entity == inner_entity or valid_entities[outer_entity].is_reference:
                    continue
                #Check if entities refer to one another
                if inner_entity.strip().strip('\'s') in outer_entity:
                    valid_entities[outer_entity].referred_by = inner_entity
                    valid_entities[inner_entity].refers_to = outer_entity
                    valid_entities[inner_entity].is_reference = True

        #set entity dicts for each type
        # self.loc_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'LOC' or entity[1].label == 'GPE' or entity[1].label == 'LANGUAGE'}
        # self.money_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'MONEY'}
        # self.acquired_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'ORG' or entity[1].label == 'FACILITY'}
        # self.buyer_seller_entities = {entity[0]: entity[1] for entity in valid_entities.items() if entity[1].label == 'ORG' or entity[1].label == 'PERSON'}
        return valid_entities

    def process_doc(self):
        self.nlp = spacy_model
        self.spacy_doc = self.nlp(self.doc)
        self.bad_chunks = ['it', 'they', 'them', 'he', 'him', 'she', 'her']

        self.acquired_candidates = []
        self.acqbus_candidates = []
        self.acqloc_candidates = []
        self.drlamt_candidates = []
        self.purchaser_candidates = []
        self.seller_candidates = []
        self.status_candidates = []

        #self.all_entities = self.process_entities(doc.ents)

        sentences = [self.nlp(sentence.text) for sentence in list(self.spacy_doc.sents)[:3]]
        for i, sentence in enumerate(sentences):
            sent_ents = self.process_entities(sentence)
            sent_chunks = list(sentence.noun_chunks)
            self.acquired_candidates.extend(self.get_possible_acquired(sent_ents, sentence, i, sent_chunks))
            self.acqbus_candidates.extend(self.get_possible_acqbus(sent_ents, sentence, i, sent_chunks))
            self.acqloc_candidates.extend(self.get_possible_acqloc(sent_ents, sentence, i, sent_chunks))
            self.drlamt_candidates.extend(self.get_possible_drlamt(sent_ents, sentence, i, sent_chunks))
            self.purchaser_candidates.extend(self.get_possible_purchaser_seller(sent_ents, sentence, i, sent_chunks))
            self.seller_candidates.extend(self.get_possible_purchaser_seller(sent_ents, sentence, i, sent_chunks))
            self.status_candidates.extend(self.get_possible_status(sentence, i, sent_chunks))

        sentence = self.nlp(self.sentences[0].text)
        # chunks = [(chunk.text, chunk.root.text, chunk.root.dep_,
        #     chunk.root.head.text) for chunk in sentence.noun_chunks]

        self.chunks = list(sentence.noun_chunks)
        self.chunks_text = [chunk.text for chunk in self.chunks]

    def match_to_chunk(self, entity_text, chunks, sentence):
        #Try to match entity to chunk with exact text
        for chunk in chunks:
            if entity_text == chunk.text:
                return chunk
        #if can't match, just find the token with the same text
        indexes = get_location_sentence(sentence, entity_text)
        if indexes[0] != -1:
            return sentence[indexes[-1] - 1]

        return sentence[0]

    def get_possible_acquired(self, entities, sentence, sent_idx, chunks):
        #process acquired entities here
        #maybe exclude certain noun chunks that won't be acquired like 'they' etc.
        acquired_candidates = []
        used_chunks = list(entities.keys())
        for item in entities.items():
            if item[1].label == 'FACILITY' or item[1].label == 'FAC' or item[1].label == 'ORG' or item[1].label == 'PRODUCT' \
                 or item[1].label == 'WORK_OF_ART':
                chunk = self.match_to_chunk(item[1].text, chunks, sentence)
                acquired_candidates.append(FeatureExtractor(chunk, self.doc, sentence, sent_idx, entity=item[1]))
                used_chunks.append(chunk.text)
        for chunk in chunks:
            if chunk.text not in used_chunks and chunk.text not in self.bad_chunks:
                acquired_candidates.append(FeatureExtractor(chunk, self.doc, sentence, sent_idx))
        
        return acquired_candidates

    def get_possible_acqbus(self, entities, sentence, sent_idx, chunks):
        acbus_candidates = []
        used_chunks = list(entities.keys())

        for chunk in chunks:
            if chunk.text not in used_chunks and chunk.text not in self.bad_chunks:
                feature_vec = FeatureExtractor(chunk, self.doc, sentence, sent_idx)
                acbus_candidates.append(feature_vec)

        return acbus_candidates
        

    def get_possible_acqloc(self, entities, sentence, sent_idx, chunks):
        #include noun chunks or naw?
        #put the location concatenation code here
        used_chunks = list(entities.keys())
        loc_candidates = []
        for item in entities.items():
            if item[1].label == 'GPE' or item[1].label == 'LOC' or item[1].label == 'LANGUAGE' or item[1].label == 'NORP': #NORP is a maybe
                chunk = self.match_to_chunk(item[1].text, chunks, sentence)
                loc_candidates.append(FeatureExtractor(chunk, self.doc, sentence, sent_idx, entity=item[1]))
                used_chunks.append(chunk.text)

        return loc_candidates

    def get_possible_drlamt(self, entities, sentence, sent_idx, chunks):
        #include dlramt processing here, including adding 'undisclosed' etc
        used_chunks = list(entities.keys())
        drlamt_candidates = []
        for item in entities.items():
            if item[1].label == 'MONEY':
                chunk = self.match_to_chunk(item[1].text, chunks, sentence)
                drlamt_candidates.append(FeatureExtractor(chunk, self.doc, sentence, sent_idx, entity=item[1]))
                used_chunks.append(chunk.text)

        return drlamt_candidates

    def get_possible_purchaser_seller(self, entities, sentence, sent_idx, chunks):
        #process purchaser and seller entities here
        used_chunks = list(entities.keys())
        purchaser_seller_candidates = []
        for item in entities.items():
            if item[1].label == 'PERSON' or item[1].label == 'ORG' or item[1].label == 'PER' or item[1].label == 'NORP':
                chunk = self.match_to_chunk(item[1].text, chunks, sentence)
                purchaser_seller_candidates.append(FeatureExtractor(chunk, self.doc, sentence, sent_idx, entity=item[1]))
                used_chunks.append(chunk.text)

        return purchaser_seller_candidates

    def get_possible_status(self, sentence, sent_idx, chunks):
        #some testing might be good here, could use other noun phrases or try to get verb phrases
        loc_candidates = []
        with open('all_statuses.json') as statuses_json:
            status_dict = json.load(statuses_json)

        seen_statuses = []
        for status in status_dict:
            if status in sentence.text:
                seen_statuses.append(status)
        
        for status in seen_statuses:
            chunk = self.match_to_chunk(status, chunks, sentence)
            status_ent = type('obj', (object,), {'text': status, 'label': 'nonent'})
            loc_candidates.append(FeatureExtractor(chunk, self.doc, sentence, sent_idx, entity=status_ent))
        return loc_candidates

    def get_acquired(self):
        for entity in self.entities:
            if entity.label == 'ORG' or entity.label == 'FACILITY':
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
    test_doc = TestDocument("./data/docs/19683")
    features = test_doc.get_feature_vector()
    print(features)