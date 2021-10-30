# text_classification_mlops

## python libs
```
python -m pip install -r requirements.txt
```
## prepare environment
```
mkdir data/
mkdir data/data_split/
mkdir model/
mkdir test_preds/
mkdir plots/
mkdir terminal_logs/
mkdir glove/
```
```
cd glove
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B/glove.6B.300d.txt ./
rm -r glove.6B
rm glove.6B.zip
```
## run streamlit app
```
streamlit run st_app.py
```
