import gdown

URL = 'https://drive.google.com/file/d/1NEduGeDHbP0sBxcV-X5Wer0QVa56bWW2/view?usp=sharing'
OUTPUT = 'params.csv'
gdown.download(URL, OUTPUT, quiet=False, fuzzy=True)