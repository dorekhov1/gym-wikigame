mkdir data
cd data || exit
wget https://dumps.wikimedia.org/enwiki/20200901/enwiki-20200901-pages-articles-multistream.xml.bz2
mkdir extracted
python -m wikiextractor.WikiExtractor --json -l -o extracted enwiki-20200901-pages-articles-multistream.xml.bz2
cd .. || exit
python data_extractor.py
