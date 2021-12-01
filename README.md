# corporate-acquisition-event-extraction-nlp
This is a system to extract events and roles from a text corpus of corporate acquisitions.

** FOR TA GRADING **
Our project is already installed onto Aspen's cade machine in
/home/u1175792/cs5340/Project/corporate-acquisition-event-extraction-nlp
within this folder is our virtual environment called venv which contains all installed libraries. We have set the permissions on this folder to 755. To run our program first activate the virtual environment using the command: 
source venv/bin/activate.csh
Once the environment is activated you can run the program with the following command:
python3 extract.py <doclist> 

a) For this project we used SpaCy with the pretrained en_core_web_trf pipeline - https://spacy.io/
We also loaded in the GloVe wikipedia 100 dimension pretrained word embedding model using Gensim https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html
To create our classifier we used sklearn's GradientBoostClassifier https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier.predict

b) It should take about a minute to load in our models initially and then each document should take about 1-2 seconds

c) Gabrielius Kudirka primarily worked on creating the machine learning classifier and implementing it into our system. The data_extractor_train file was created by 
Gabrielius and extracts all potential candidates for classification as well as all of their features and their label. Once these are extracted from the text they
are formatted as a numerical vector using one hot encoding and word embeddings to be inputted into our classifier. Gabrielius also created and tuned the GradientBoostingClassifier in the ml_classifier.py and ml_classifier.ipynb files. Gabrielius also worked on extracting candidates for the slots and formatting them
for the classifier at test time.

Aspen Evans worked on formatting the extracted potential slot candidates so they would more accurately reflect what the expected outputs for these slots would be.
In particular she worked on creating new money entities for occurences 'undisclosed' in the text and concatenating comma separatedpotential locations that would come up as separated from spaCy's NER system. These processed entities would be inputted into Gabrielius's code to be classified by our classifier.

d) Our program produces a large number of warnings while running however these can be disregarded, an output template file should still be created at the end
















HOW TO RUN THIS PROJECT FROM GIT:
1. Navigate to your desired directory and clone our project by running:
git clone https://github.com/gabekudirka/corporate-acquisition-event-extraction-nlp.git

2. cd into the cloned directory:
cd corporate-acquisition-event-extraction-nlp

3. Create a local virtual environment:
python3 -m venv venv

4. Activate the virtual environment:
source venv/Scripts/activate

5. Install dependencies:
pip install -r requirements.txt

6. Run the program:
python3 extract.py <doclist>

7. When finished, deactivate the environment
deactivate