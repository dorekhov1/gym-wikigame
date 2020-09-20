# gym-wikigame

## Setup Python 3.8

Install Python libraries:


`pip install -r requirements.txt`

## Setup raw data

To get the raw Wikipedia data dump (17.4 GB) and process it, run the following script:

`bash data_setup.sh`

This will create a `wiki.pickle` file in the `data` directory. The file contains a Python dictionary, which maps Wikipedia page titles to all relevant data of the page, including a list of internal links in the page.

If you do not want to download and preprocess the entire dump, I will soon try to find a way to host the `wiki.pickle` file somewhere, so it's downloadable.