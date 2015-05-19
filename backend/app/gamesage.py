import string
import gensim
from nltk import WordNetLemmatizer


class GameSage(object):
    """An anthropomorphization of the procedure in LSA called 'folding in'."""

    def __init__(self, database, term_id_dictionary, tf_idf_model, lsa_model, user_submitted_text):
        """Initialize a GameSage object."""
        self.database = database
        self.term_id_dictionary = term_id_dictionary
        self.tf_idf_model = tf_idf_model
        self.lsa_model = lsa_model
        preprocessed_text = self._preprocess_text(text=user_submitted_text)
        lsa_vector_for_user_submitted_text = self._fold_in_user_submitted_text(text=preprocessed_text)
        self.most_related_games, self.least_related_games = self._get_most_related_games_to_user_submitted_text(
            lsa_vector_for_user_submitted_text=lsa_vector_for_user_submitted_text
        )
        self.most_related_games_str, self.least_related_games_str = (
            self._generate_related_games_strings()
        )

    def _generate_related_games_strings(self):
        """Generate strings representing the most and least related games, for GameNet to parse."""
        most_related_games_str = (
            ','.join('{}&{}'.format(entry[0], entry[1]) for entry in self.most_related_games)
        )
        least_related_games_str = (
            ','.join('{}&{}'.format(entry[0], entry[1]) for entry in self.least_related_games)
        )
        return most_related_games_str, least_related_games_str

    def _get_most_related_games_to_user_submitted_text(self, lsa_vector_for_user_submitted_text):
        """Get the 50 most related and unrelated games to the user-submitted text."""
        # Reindex the LSA space to account for the folding in (ignore first dimension of
        # the LSA vector for the user-submitted text)
        corpus_including_new_lsa_vector = (
            [game.lsa_vector for game in self.database] + [lsa_vector_for_user_submitted_text]
        )
        reindexed_lsa_space = gensim.similarities.docsim.Similarity(
            output_prefix="temp_gensim_lsa_file_that_can_be_deleted",
            corpus=corpus_including_new_lsa_vector,
            num_features=207, num_best=len(corpus_including_new_lsa_vector)
        )
        lsa_scores_for_all_games_relative_to_this_game = (
            reindexed_lsa_space[lsa_vector_for_user_submitted_text]
        )
        most_related_games = lsa_scores_for_all_games_relative_to_this_game[1:51]  # [0] will be the text itself
        least_related_games = lsa_scores_for_all_games_relative_to_this_game[-50:]
        least_related_games.reverse()  # Order these with least related game first
        return most_related_games, least_related_games

    def _fold_in_user_submitted_text(self, text):
        """Fold user-submitted text into our LSA model, i.e., derive an LSA vector for the text."""
        frequency_count_vector_for_user_submitted_text = (
            (self.term_id_dictionary.doc2bow(text.split()))
        )
        tf_idf_vector_for_user_submitted_text = (
            self.tf_idf_model[frequency_count_vector_for_user_submitted_text]
        )
        document_lsa_vector_for_user_submitted_text = (
            self.lsa_model[tf_idf_vector_for_user_submitted_text]
        )
        # Exclude first dimension, as we've already done with the existing LSA vectors
        lsa_vector_for_user_submitted_text = document_lsa_vector_for_user_submitted_text[1:]
        return lsa_vector_for_user_submitted_text

    def _preprocess_text(self, text):
        """Preprocess user-submitted text in the same way we preprocessed our corpus."""
        # Remove weird characters that could cause encoding issues
        text = filter(lambda char: char in string.printable, text)
        # Remove newline and tab characters
        for special_char in ('\n', '\r', '\t'):
            text = text.replace(special_char, ' ')
        # Remove preliminary set of punctuation symbols
        for punctuation_symbol in ('_', '.', ',', ';'):
            text = text.replace(punctuation_symbol, ' ')
        text = text.lower()
        # Remove redundant whitespace
        text = ' '.join(text.split())
        # Tokenize multiword game titles
        text = self._tokenize_multiword_titles(text=text)
        # Tokenize multiword platform names
        text = self._tokenize_multiword_platform_names(text=text)
        # Remove punctuation and symbols (except underscores)
        text = self._remove_punctuation_and_symbols(text=text)
        # Again remove redundant whitespace
        text = ' '.join(text.split())
        # Remove stopwords
        text = self._remove_stopwords(text=text)
        # Lemmatize words
        text = self._lemmatize_words(text=text)
        # Remove stopwords again (some may have been reintroduced
        # by lemmatization)
        text = self._remove_stopwords(text=text)
        return text

    def _tokenize_multiword_titles(self, text):
        """Tokenize occurrences of multiword titles."""
        titles = [game.title.lower() for game in self.database if game.title]
        multiword_titles = [title for title in titles if len(title.split()) > 1]
        tokenized_multiword_titles = {}
        for multiword_title in multiword_titles:
            tokenized_multiword_titles[multiword_title] = '_'.join(multiword_title.split())
        multiword_titles.sort(key=lambda t: len(t.split()), reverse=True)
        titles_sorted_by_number_of_words = multiword_titles
        text = ' {} '.format(text)
        for title in titles_sorted_by_number_of_words:
            tokenized_title = tokenized_multiword_titles[title]
            try:
                while ' {} '.format(title) in text:
                    text = text.replace(
                        ' {} '.format(title), ' {} '.format(tokenized_title)
                    )
            except UnicodeEncodeError:
                pass  # Not worth struggling with game titles with weird encodings
        text = ' '.join(text.split())
        return text

    @staticmethod
    def _tokenize_multiword_platform_names(text):
        """Tokenize occurrences of multiword platform names."""
        f = open('./static/multiword_platform_names.txt', 'r')
        multiword_platform_names = f.readlines()
        multiword_platform_names = [name.strip('\n') for name in multiword_platform_names]
        multiword_platform_names = [name.lower() for name in multiword_platform_names]
        tokenized_multiword_names = {}
        for multiword_name in multiword_platform_names:
            tokenized_multiword_names[multiword_name] = '_'.join(multiword_name.split())
        multiword_platform_names.sort(key=lambda t: len(t.split()), reverse=True)
        platform_names_sorted_by_number_of_words = multiword_platform_names
        text = ' {} '.format(text)
        for platform_name in platform_names_sorted_by_number_of_words:
            tokenized_title = tokenized_multiword_names[platform_name]
            while ' {} '.format(platform_name) in text:
                text = text.replace(
                    ' {} '.format(platform_name), ' {} '.format(tokenized_title)
                )
        return text

    @staticmethod
    def _remove_punctuation_and_symbols(text):
        """Remove punctuation and other symbols."""
        for symbol in (
            '[', ']', '\'', '"', ':', '&', '(', ')', '\\', '/', '*', '!',
            '?', '$', '^', '~', '+', '=', '{', '}', '`', '|', '#'
        ):
            text = text.replace(symbol, ' ')
        return text

    @staticmethod
    def _remove_stopwords(text):
        """Remove all stopwords from the text."""
        f = open('./static/stopwords.txt', 'r')
        stopwords = f.readlines()
        stopwords = (stopword.strip('\n') for stopword in stopwords)
        tokens = [token.lower() for token in text.split()]
        for i in xrange(len(tokens)):
            if tokens[i] in stopwords:
                tokens[i] = ''
            elif len(tokens[i]) == 1:  # Remove single letters
                tokens[i] = ''
        text = ' '.join([token for token in tokens if token])
        return text

    @staticmethod
    def _lemmatize_words(text):
        """Lemmatize all words in the text."""
        lemmatizer = WordNetLemmatizer()
        lemmatizations = {}
        tokens = text.split()
        for word in tokens:
            if word not in lemmatizations:
                lemmatizations[word] = lemmatizer.lemmatize(word)
        for i in xrange(5):  # Need to repeat several times to be safe
            tokens = text.split()
            for j in xrange(len(tokens)):
                try:
                    tokens[j] = lemmatizations[tokens[j]]
                except KeyError:
                    # During last pass, words were turned into their lemmas, which don't
                    # have entries in lemmatizations
                    pass
        text = ' '.join(tokens)
        return text