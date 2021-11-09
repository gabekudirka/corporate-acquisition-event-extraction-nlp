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
        self.acquired = ans_raw[1][10:].strip().strip('"')
        self.acqbus = ans_raw[2][8:].strip().strip('"')
        self.acqloc = ans_raw[3][8:].strip().strip('"')
        self.drlamt = ans_raw[4][8:].strip().strip('"')
        self.purchaser = ans_raw[5][11:].strip().strip('"')
        self.seller = ans_raw[6][8:].strip().strip('"')
        self.status = ans_raw[7][8:].strip().strip('"')

    def get_status(self):
        return self.status