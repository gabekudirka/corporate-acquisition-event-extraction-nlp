import spacy
from spacy.tokens.doc import Doc
from data_extractor_test import Entity
import string
from functools import cache
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import csv

class FeatureVector:
    def __init__(self, text, dep, pos, head_text, head_pos, head_dep, next_verb):
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

    def get_feature_vector(self):
        feature_vecs_output = []
        for feature_vec in self.feature_vectors:
            feature_vec_output = [feature_vec.text, feature_vec.dep, feature_vec.pos, feature_vec.head_text, feature_vec.head_pos, feature_vec.head_dep, feature_vec.next_verb,
                                  feature_vec.entity_label, feature_vec.frequency_in_doc, feature_vec.sentence_loc, feature_vec.sentence_num, feature_vec.label] 
            feature_vecs_output.append(feature_vec_output)
        return feature_vecs_output

    def get_location_sentence(self, sentence, phrase_tokens):
        for i in range(len(sentence)):
            if sentence.text.split()[i:i+len(phrase_tokens)] == phrase_tokens:
                return (i, i+len(phrase_tokens))
        return (-1, -1)

    def get_sliding_window(self, sentence, feature_vec):
        pass

    def one_hot_encode(self, feature_vec):
        POS_categories = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        dep_categories = ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']
        label_categories = ['acquired', 'acqbus', 'acqloc', 'drlamt', 'purchaser', 'seller', 'status', 'none']
        entity_categories = ['nonent', 'PERSON', 'NORP', 'FACILITY', 'FAC', 'ORG', '']
        one_hot_encoder = OneHotEncoder(sparse=False)
        one_hot_encoded = one_hot_encoder.fit_transform([feature_vec])
        return one_hot_encoded

    def get_ml_features(self):
        nlp = spacy.load("en_core_web_trf")
        self.spacy_doc = nlp(self.doc)
        self.entities = self.process_entities(self.spacy_doc.ents)

        sentences = [nlp(sentence.text) for sentence in list(self.spacy_doc.sents)[:3]]
        feature_vectors = []
        sentences_to_consider = min(len(sentences) - 1, 3)
        for i in range(sentences_to_consider):
            feature_vectors.append(self.process_chunks(sentences[i].noun_chunks, sentences[i]))
            for feature_vec in feature_vectors[i]:
                feature_vec.sentence_num = i
                feature_vec.sentence_loc = self.get_location_sentence(sentences[i], feature_vec.tokens)[0]
                if feature_vec.text in self.entities.keys() and self.entities[feature_vec.text].is_reference == False:
                    feature_vec.entity_label = self.entities[feature_vec.text].label
                else:
                    feature_vec.entity_label = 'nonent'
                feature_vec.frequency_in_doc = (self.doc.count(feature_vec.text))

        self.feature_vectors = [vector for sublist in feature_vectors for vector in sublist]

    def process_chunks(self, chunks, sentence):
        tokenized_true_slots = {slot[1]: (slot[0], []) for slot in self.true_slots.items()}
        feature_vector_list = []
        #This loop finds all found noun phrases that are like the gold slots
        #Adds noun phrases that are not like the gold slots to the feature vector list
        chunk_list = list(chunks)
        for chunk in chunk_list:
            if chunk.text == 'Reuter':
                continue
            feature_vec = FeatureVector(chunk.text, chunk.root.dep_, chunk.root.pos_, chunk.root.head.text, chunk.root.head.pos_, chunk.root.head.dep_, self.get_next_verb(chunk, sentence))
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
            slot_entity =  tokenized_true_slots[slot][0]
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
            slot_entity_tokens = slot_entity.split()
            indexes = self.get_location_sentence(sentence, slot_entity_tokens)
            if indexes[0] != -1:
                token = sentence[indexes[-1] - 1]
                #next verb is not correct
                feature_vec = FeatureVector(tokenized_true_slots[slot][0], token.dep_, token.pos_, token.head.text, token.head.pos_, token.head.dep_, self.get_next_verb(token, sentence, ischunk=False))
                feature_vec.label = slot
                feature_vector_list.append(feature_vec)
            #If we can't find the gold slot manually (for some reason), use a related noun phrase vector with gold phrase as text
            else:
                for feature_vec in tokenized_true_slots[slot][1]:
                    if feature_vec.text in self.entities.keys() and self.entities[feature_vec.text].is_reference == True:
                        continue
                    else:
                        feature_vec = tokenized_true_slots[slot][1][0]
                        feature_vec.label = slot
                        feature_vec.text = tokenized_true_slots[slot][0]
                        feature_vector_list.append(feature_vec)
                        break
                
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

    #Eventually make this recursive - use @cache
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
    test_doc = TrainDocument("./data/docs/11996", "./data/anskeys/11996.key", get_ml_features=True)
    features = test_doc.get_feature_vector()
    with open('output.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(features)