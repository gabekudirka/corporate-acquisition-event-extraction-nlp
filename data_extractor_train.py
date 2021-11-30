import spacy
from spacy.tokens.doc import Doc
from data_extractor_test import Entity
import string
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import csv
import gensim.downloader
import numpy as np
import re

class FeatureVector:
    def __init__(self, text, dep, pos, head_text, head_pos, head_dep, next_verb, sentence):
        self.text = text
        self.tokens = text.split()
        self.dep = dep
        if len(self.tokens) > 0 and self.tokens[0][0].isupper() and self.tokens[-1][0].isupper():
            self.pos = 'PROPN'
        else:
            self.pos = pos
        self.head_text = head_text
        self.head_pos = head_pos
        self.head_dep = head_dep
        self.next_verb = next_verb
        self.label = 'none'
        self.add_to_list = True
        self.sentence = sentence

spacy_model = spacy.load('en_core_web_trf')
gensim_model = gensim.downloader.load('glove-wiki-gigaword-50')

class TrainDocument:
    def __init__ (self, doc_filepath, ans_filepath, get_ml_features=False):
        self.doc = self.read_doc(doc_filepath)
        self.read_ans(ans_filepath)

        if get_ml_features:
            #trf is most accurate ner model in spacy, doesn't include word embeddings
            self.nlp = spacy_model
            #load in the md pipeline just for word embeddings
            #self.nlp_md = spacy.load('en_core_web_md', exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
            self.gensim_model = gensim_model
            self.spacy_doc = self.nlp(self.doc)
            self.get_ml_features()

    def read_doc(self, filepath):
        text_raw = open(filepath, "r").read()
        return text_raw.replace('\n', ' ').strip();

    def read_ans(self, filepath):
        ans_raw = open(filepath, "r").readlines()
        self.true_slots = {}

        for line in ans_raw:
            line_tokens = line.split()
            if len(line_tokens) <= 0:
                continue
            if line_tokens[0] == 'TEXT:':
                self.text = ans_raw[0][6:].strip()
            elif line_tokens[0] == 'ACQUIRED:':
                entities = re.findall('"([^"]*)"', line)
                if len(entities) == 0:
                    self.acqbus = '---'
                    continue
                for entity in entities:
                    self.true_slots[entity] = 'acquired'
                self.acquired = entities[0]
            elif line_tokens[0] == 'ACQBUS:':
                entities = re.findall('"([^"]*)"', line)
                if len(entities) == 0:
                    self.acqbus = '---'
                    continue
                for entity in entities:
                    self.true_slots[entity] = 'acqbus'
                self.acqbus = entities[0]
            elif line_tokens[0] == 'ACQLOC:':
                entities = re.findall('"([^"]*)"', line)
                if len(entities) == 0:
                    self.acqbus = '---'
                    continue
                for entity in entities:
                    self.true_slots[entity] = 'acqloc'
                self.acqloc = entities[0]
            elif line_tokens[0] == 'DLRAMT:':
                entities = re.findall('"([^"]*)"', line)
                if len(entities) == 0:
                    self.acqbus = '---'
                    continue
                for entity in entities:
                    self.true_slots[entity] = 'drlamt'
                self.drlamt = entities[0]
            elif line_tokens[0] == 'PURCHASER:':
                entities = re.findall('"([^"]*)"', line)
                if len(entities) == 0:
                    self.acqbus = '---'
                    continue
                for entity in entities:
                    self.true_slots[entity] = 'purchaser'
                self.purchaser = entities[0]
            elif line_tokens[0] == 'SELLER:':
                entities = re.findall('"([^"]*)"', line)
                if len(entities) == 0:
                    self.acqbus = '---'
                    continue
                for entity in entities:
                    self.true_slots[entity] = 'seller'
                self.seller = entities[0]
            elif line_tokens[0] == 'STATUS:':
                entities = re.findall('"([^"]*)"', line)
                if len(entities) == 0:
                    self.acqbus = '---'
                    continue
                for entity in entities:
                    self.true_slots[entity] = 'status'
                self.status = entities[0]

    def get_status(self):
        return self.status

    def get_feature_vector(self):
        feature_vecs_output = []
        for feature_vec in self.feature_vectors:
            feature_vec_output = [feature_vec.label, feature_vec.dep, feature_vec.pos, feature_vec.head_pos, feature_vec.head_dep, 
                                  feature_vec.entity_label, feature_vec.frequency_in_doc, feature_vec.sentence_loc, feature_vec.sentence_num]

            feature_vec_output.extend(self.get_word_embeddings(feature_vec.text))
            feature_vec_output.extend(self.get_word_embeddings(feature_vec.head_text))
            feature_vec_output.extend(self.get_word_embeddings(feature_vec.next_verb))
            
            for word in feature_vec.sliding_window:
                feature_vec_output.extend(self.get_word_embeddings(word))

            feature_vecs_output.append(feature_vec_output)
        return feature_vecs_output

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

        for feature_vec in self.feature_vectors:
            feature_vec.dep_one_hot = np.zeros(len(possible_dep_arr)).tolist()
            feature_vec.head_pos_one_hot = np.zeros(len(possible_pos_arr)).tolist()
            feature_vec.head_dep_one_hot = np.zeros(len(possible_dep_arr)).tolist()
            feature_vec.ent_label_one_hot = np.zeros(len(possible_ent_labels_arr)).tolist()

            if self.dep in possible_dep:
                self.dep_one_hot[possible_dep[self.dep]] = 1
            if self.head_pos in possible_pos:
                self.head_pos_one_hot[possible_pos[self.head_pos]] = 1
            if self.head_dep in possible_dep:
                self.head_dep_one_hot[possible_dep[self.head_dep]] = 1
            if self.entity_label in possible_ent_labels:
                self.ent_label_one_hot[possible_ent_labels[self.entity_label]] = 1

    def get_location_sentence(self, sentence, phrase):
        try:
            start_index = sentence.text.index(phrase)
            start_loc = len(sentence.text[:start_index].split())
            return (start_loc, start_loc + len(phrase.split()))
        except:
            return (-1, -1)
        # phrase_tokens = phrase.split()
        # for i in range(len(sentence)):
        #     if sentence.text.split()[i:i+len(phrase_tokens)] == phrase_tokens:
        #         return (i, i+len(phrase_tokens))
        # return (-1, -1)

    #get two word sliding window
    def get_sliding_window(self, feature_vec):
        phrase = feature_vec.text
        sentence = feature_vec.sentence
        sentence_tokens = sentence.text.split()
        indexes = self.get_location_sentence(sentence, phrase)

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

    def get_ml_features(self):
        
        self.entities = self.process_entities(self.spacy_doc.ents)

        sentences = [self.nlp(sentence.text) for sentence in list(self.spacy_doc.sents)[:3]]
        feature_vectors = []
        sentences_to_consider = min(len(sentences), 3)
        for i in range(sentences_to_consider):
            feature_vectors.append(self.process_chunks(sentences[i].noun_chunks, sentences[i]))
            for feature_vec in feature_vectors[i]:
                feature_vec.sentence_num = i
                feature_vec.sentence_loc, _ = self.get_location_sentence(sentences[i], feature_vec.text)
                if feature_vec.text in self.entities.keys() and self.entities[feature_vec.text].is_reference == False:
                    feature_vec.entity_label = self.entities[feature_vec.text].label
                else:
                    feature_vec.entity_label = 'nonent'
                feature_vec.frequency_in_doc = (self.doc.count(feature_vec.text))
                feature_vec.sliding_window = self.get_sliding_window(feature_vec)

        self.feature_vectors = [vector for sublist in feature_vectors for vector in sublist]
        self.one_hot_encode()
        x = 1

    def process_chunks(self, chunks, sentence):
        tokenized_true_slots = {}
        for item in self.true_slots.items():
            if item[1] in tokenized_true_slots.keys():
                tokenized_true_slots[item[1]][0].append(item[0])
                continue
            tokenized_true_slots[item[1]] = ([item[0]], [])

        feature_vector_list = []
        #This loop finds all found noun phrases that are like the gold slots
        #Adds noun phrases that are not like the gold slots to the feature vector list
        chunk_list = list(chunks)
        for chunk in chunk_list:
            if chunk.text == 'Reuter':
                continue
            feature_vec = FeatureVector(chunk.text, chunk.root.dep_, chunk.root.pos_, chunk.root.head.text, chunk.root.head.pos_,
                                        chunk.root.head.dep_, self.get_next_verb(chunk, sentence), sentence)
            for entity in self.true_slots.keys():
                entity_tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in entity.split()]
                if all(token in feature_vec.tokens for token in entity_tokens):
                    tokenized_true_slots[self.true_slots[entity]][1].append(feature_vec)
                    feature_vec.add_to_list = False
                elif all(token in entity_tokens for token in feature_vec.tokens):
                    tokenized_true_slots[self.true_slots[entity]][1].append(feature_vec)
                    feature_vec.add_to_list = False
            if feature_vec.add_to_list:
                feature_vector_list.append(feature_vec)
            
        #Make sure that all of the gold slots are added to the list with labels
        for slot in tokenized_true_slots.keys():
            slot_entities =  tokenized_true_slots[slot][0]
            for slot_entity in slot_entities:
                if slot_entity == '---':
                    continue
                
                #If a noun phrase is found with the exact same text as gold slot, set label and add it to list
                gold_vec = self.match_with_gold_text(slot_entity, tokenized_true_slots[slot][1])
                if gold_vec != None:
                    gold_vec.label = slot
                    feature_vector_list.append(gold_vec)
                    continue
                        
                #If no noun phrase is found with the exact same text as gold slot, find the gold slot in the doc manually
                #and add it to the feature list
                indexes = self.get_location_sentence(sentence, slot_entity)
                if indexes[0] != -1:
                    token = sentence[indexes[-1] - 1]
                    feature_vec = FeatureVector(slot_entity, token.dep_, token.pos_, token.head.text, token.head.pos_, 
                                                token.head.dep_, self.get_next_verb(token, sentence, ischunk=False), sentence)
                    feature_vec.label = slot
                    feature_vector_list.append(feature_vec)
                #If we can't find the gold slot manually (for some reason), use a related noun phrase vector with gold phrase as text
                # else:
                #     for feature_vec in tokenized_true_slots[slot][1]:
                #         if feature_vec.text in self.entities.keys() and self.entities[feature_vec.text].is_reference == True:
                #             continue
                #         else:
                #             feature_vec = tokenized_true_slots[slot][1][0]
                #             feature_vec.label = slot
                #             feature_vec.text = slot_entity
                #             feature_vector_list.append(feature_vec)
                #             break
                
        return feature_vector_list

    def match_with_gold_text(self, slot_entity, vec_list):
        for feature_vec in vec_list:
                if feature_vec.text == slot_entity:
                    return feature_vec
        return None

    def match_chunk_to_label(self, chunk):
        for entity in self.true_slots.keys():
            if entity in chunk.text:
                return self.true_slots[entity]
            if chunk.text in entity:
                return self.true_slots[entity]
        return 'none'

    def get_next_verb(self, target, sentence, ischunk=True):
        if ischunk:
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
    test_doc = TrainDocument("./data/docs/2434", "./data/anskeys/2434.key", get_ml_features=True)
    features = test_doc.get_feature_vector()
    with open('output_test.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(features)