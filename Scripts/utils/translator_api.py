'''
WARNING: googletrans is now not working well, please use tranlate-shell instead.
'''
import pandas as pd
import googletrans
import pprint
from googletrans import Translator
translator = Translator()

pp = pprint.PrettyPrinter(indent=4)


# pd.set_option('max_colwidth', 300)

# how to get the supported language and their corresponing code
# lang_df = pd.DataFrame.from_dict(googletrans.LANGUAGES,  orient='index', columns=['Language'])
# print(lang_df.index.tolist())


result = translator.translate(text='Hello world', dest='es', src='en')
# pp.pprint(result.__dict__())
print(result.text)