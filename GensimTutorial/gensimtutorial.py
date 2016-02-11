import _pickle
import glob
import os
from gensim import corpora, models, similarities, parsing

class Main():
    current_working_directory = os.getcwd()
    licenses_path = current_working_directory + '\Licenses\*.txt'
    corpora_dict_filepath = current_working_directory + '\Models\corpora.dict'
    corpus_filepath = current_working_directory + '\Models\corpus.mm'
    lsi_filepath = current_working_directory + '\Models\lsimodel'
    similarity_index_filepath = current_working_directory + '\Models\lsi_sim.index'
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

    def create_separate_list_of_license_names_and_content(self, licenses_dict_tokenized):
        tokenized_content_list = list()
        license_name_list = list()
        for licence_name, tokens in licenses_dict_tokenized.items():
            license_name_list.append(licence_name)
            tokenized_content_list.append(tokens)
        return license_name_list, tokenized_content_list

    def create_id_word_dictionary(self, tokenized_content_list):
        corpora_dict = corpora.Dictionary(tokenized_content_list)
        corpora_dict.save(self.corpora_dict_filepath)
        return corpora_dict

    def create_marketmatrix_corpus(self, corpora_dict, tokenized_content_list):
        corpus = list()
        for tokens in tokenized_content_list:
            corpus.append(corpora_dict.doc2bow(tokens))
        corpora.MmCorpus.serialize(self.corpus_filepath, corpus)

    def load_corpora_dict(self):
        corpora_dict = corpora.Dictionary.load(self.corpora_dict_filepath)
        return corpora_dict

    def load_corpus(self):
        corpus = corpora.MmCorpus(self.corpus_filepath)
        return corpus

    def create_lsi_model(self, corpus, corpora_dict):
        lsi_model = models.LsiModel(corpus, id2word=corpora_dict, num_topics=50)
        lsi_model.save(self.lsi_filepath)
        return lsi_model

    def create_lsi_similarity_index(self, lsi_model, corpus):
        similarity_index = similarities.MatrixSimilarity(lsi_model[corpus])
        similarity_index.save(self.similarity_index_filepath)
        return similarity_index
        
    def load_lsi_model(self):
        lsi_model = models.LsiModel.load(self.lsi_filepath)
        return lsi_model

    def load_lsi_similarity_index(self):
        similarity_index = similarities.MatrixSimilarity.load(self.similarity_index_filepath)
        return similarity_index

    def create_query(self, query_content_filepath, corpora_dict, lsi_model):
        query_content = self.get_file_content(query_content_filepath)
        tokenized_content = parsing.preprocess_string(query_content)
        vec_bag_of_words = corpora_dict.doc2bow(tokenized_content)
        vec_lsi = lsi_model[vec_bag_of_words]
        return vec_lsi

    def perform_similarity_query(self, similarity_index, query_lsi):
        similarity_scores = similarity_index[query_lsi]
        return similarity_scores
        
    def match_license_to_score(self, similarity_scores, license_name_list):
        license_score_dict = dict()
        for index, score in enumerate(similarity_scores):
            license = license_name_list[index]
            formatted_score = score #Put math here to mess with score if you want something more readable than a coefficient.
            license_score_dict[license] = formatted_score
        return license_score_dict

    def sort_score_dict(self, license_score_dict):
        results = sorted(license_score_dict.items(), key=lambda x:float(x[1]), reverse=True)
        return results

    def top_result_rules_output(self, sorted_results, rules_dict):
        license_name, score = sorted_results[0]
        print(license_name)
        print(rules_dict[license_name])


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
    license_name_list, tokenized_content_list = main.create_separate_list_of_license_names_and_content(licenses_dict)
    # Serialize the license name list with pickle to load later if reusing with corpus. I am skipping this because this script will run with it once in memory.
    # You will also want to save the 'rules_dict' as well which is utilized at the very end for output.

    # Step 3: Create corpora dictionary where a unique id is assigned to each token/word and the corpus
    corpora_dict = main.create_id_word_dictionary(tokenized_content_list)
    main.create_marketmatrix_corpus(corpora_dict, tokenized_content_list)
    # Load newly created corpus from file
    corpus = main.load_corpus()
    
    # Step 4: Transform the bag-of-word corpus into a latent semantic index model.
    lsi_model = main.create_lsi_model(corpus, corpora_dict)
    # Create similarity index for lsi_model
    similarity_index = main.create_lsi_similarity_index(lsi_model, corpus)
        
    # Everything is now ready to perform a similarity query.
    # Corpus and lsi_model can be loaded and reused for future use. They do not have to be recreated unless
    # you decide to retrain/add licenses to the corpus. Also, you can just add new documents to the previously trained model and update it without recreation.
    # However, it is advisable to recreate because you the the license_name list to match up exactly as here the index is used to match the name with the similarity score index.

    # Step 5: Read content for query from file and prepare the query 
    # I took a snippet of the apache license as the test query.
    query_filepath = os.getcwd() + '\\test_query_apache.txt'
    query_coordinates_in_lsi_space = main.create_query(query_filepath, corpora_dict, lsi_model)

    # Step 6: Perform the similarity query
    similarity_scores = main.perform_similarity_query(similarity_index, query_coordinates_in_lsi_space)
    license_score_dict = main.match_license_to_score(similarity_scores, license_name_list)
    results = main.sort_score_dict(license_score_dict)

    print("Here are the results printed as is: ")
    print(results)
    print("\nTop result matched to rules with print out: ")
    main.top_result_rules_output(results, rules_dict)
    
     