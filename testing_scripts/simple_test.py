import spacy
import en_core_web_trf
from data_extractor_train import TrainDocument
from data_extractor_test import  TestDocument
from file_data_extractor_train import extract_files_train
from file_data_extractor_test  import extract_files_test

all_docs_test = extract_files_test()

for doc in all_docs_test:
    print("TEXT: ", doc.get_text(doc.get_filepath()))
    print("AQUIIRED: ", doc.get_acquired())
    print("ACQBUS: ---")
    print("ACQLOC: ", doc.get_acqloc())
    print("DLRAMT: ", doc.get_drlamt())
    print("PURCHASER: ", doc.get_purchaser())
    print("SELLLER: ", doc.get_seller())
    doc.set_status()
    print("STATUS: ", doc.get_status())