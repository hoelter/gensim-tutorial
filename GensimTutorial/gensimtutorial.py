import _pickle
import glob
import os
from gensim import corpora, models, similarities, parsing

class Main():
    current_working_directory = os.getcwd()
    licenses_path = current_working_directory + '\Licenses\*.txt'
    corpora_dict_filepath = current_working_directory + "corporadict.dict"
    corpus_filepath = current_working_directory + "corpus.mm"
    rules_signal = "-rules"

    def get_file_paths(self):
        file_paths = glob.iglob(self.licenses_path)
        return file_paths

    def get_file_content(self, file_path):
            with open(file_path, 'r') as file:
                content = file.read()
            return content

    def get_license_name_from_path(self, file_path):
        name = os.path.basename(file_path)
        name = os.path.splitext(name)[0]
        if self.rules_signal in name:
            name = name[0:(len(self.rules_signal) * -1)]
        return name

    def create_name_content_dict(self, file_paths):
        name_content_dict = dict()
        for path in file_paths:
            content = self.get_file_content(path)
            name = self.get_license_name_from_path(path)
            name_content_dict[name] = content
        return name_content_dict

    def separate_rules_and_licence_paths(self, file_paths):
        rule_paths = list()
        license_paths = list()
        for path in file_paths:
            if self.rules_signal in path:
                rule_paths.append(path)
            else:
                license_paths.append(path)
        return rule_paths, license_paths

    def tokenize_dictionary_content(self, licenses_dict):
        for name, content in licenses_dict.items():
            licenses_dict[name] = parsing.preprocess_string(content)

    def create_list_of_content(self, licenses_dict_tokenized):
        tokenized_content_list = list()
        for _, tokens in licenses_dict_tokenized.items():
            tokenized_content_list.append(tokens)
        return tokenized_content_list

    def create_vector_word_dictionary(self, tokenized_content_list):
        corpora_dict = corpora.Dictionary(tokenize_dictionary_content)
        corpora_dict.save(self.corpora_dict_filepath)
        return corpora_dict

    def create_marketmatrix_corpus(self, corpora_dict, tokenized_content_list):
        corpus_prep_list = list()
        for tokens in tokenized_content_list:
            corpus.append(corpora_dict.doc2bow(tokens))
        corpora.MmCorpus.serialize(self.corpus_filepath, corpus_prep_list)

    def load_corpora_dict(self):
        corpora_dict = corpora.Dictionary.load(self.corpora_dict_filepath)
        return corpora_dict

    def load_corpus(self):
        corpus = corpora.Dictionary.load(self.corpus_filepath)
        return corpus
        

        
                 
       

if __name__=="__main__":
    main = Main()

    # Step 1: read all licences and associated rules. Create dictionaries for licences and rules content with a matching key/name.
    file_paths = main.get_file_paths()
    rule_paths, licence_paths = main.separate_rules_and_licence_paths(file_paths)
    rules_dict = main.create_name_content_dict(rule_paths)
    licenses_dict = main.create_name_content_dict(licence_paths)

    # Step 2: Prep for corpus creation. Manipulate data to be able to access values after vectorizing.
    # Tokenize the license content
    main.tokenize_dictionary_content(licenses_dict)
    # Create a list of just the content to be able to acces via index
    tokenized_content_list = main.create_list_of_content(licenses_dict)
 
    # Step 3: Create corpora vector/word dictionary and the corpus
    corpora_dict = main.create_vector_word_dictionary(tokenized_content_list)
    main.create_marketmatrix_corpus(corpora_dict, tokenized_content_list)
    
     