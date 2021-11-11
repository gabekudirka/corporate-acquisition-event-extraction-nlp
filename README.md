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

b) It should take about 5 seconds to process each document.

c) Gabrielius Kudirka extracted entities with SpaCy and assigned them to the acquired, acqloc, drlamt, purchaser, and seller slots.
Aspen Evans extracted the statuses from the training documents into a list and set the status based on whether one of the extracted statuses was found in a given document.

d) We have not yet found a reliable way to find the ACQBUS slot so we are not setting this yet. Despite the long runtime though there are no known major limitations.
















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