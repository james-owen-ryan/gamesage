import csv
import gensim
from flask import Flask, render_template, jsonify, request
from gamesage import GameSage
from game import Game


app = Flask(__name__)


@app.route('/gamesage')
def home():
    """Render the GameSage homepage."""
    return render_template('gamesage_index.html')


@app.route('/gamesage/submittedText', methods=['POST'])
def generate_gamenet_query():
    """Generate a query for GameNet."""
    user_submitted_text = request.form['user_submitted_text']
    gamesage = GameSage(
        database=app.database, term_id_dictionary=app.term_id_dictionary,
        tf_idf_model=app.tf_idf_model, lsa_model=app.lsa_model,
        user_submitted_text=user_submitted_text
    )
    return jsonify(
        user_submitted_text=user_submitted_text,
        most_related_games_str=gamesage.most_related_games_str,
        least_related_games_str=gamesage.least_related_games_str
    )


def load_gamesage_database():
    """Load the database of game representations from a TSV file."""
    database = []
    with open('static/game_lsa_vectors.tsv', 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            game_id, title, year, lsa_vector_str = row
            game_object = Game(game_id, title, lsa_vector_str)
            database.append(game_object)
    return database


def load_term_id_dictionary():
    """Load the term-ID dictionary for our corpus."""
    term_id_dictionary = gensim.corpora.Dictionary.load('./static/id2term.dict')
    return term_id_dictionary


def load_tf_idf_model():
    """Load our tf-idf model."""
    tf_idf_model = (
        gensim.models.TfidfModel.load('./static/wiki_games_tfidf_model')
    )
    return tf_idf_model


def load_lsa_model():
    """Load our LSA model."""
    lsa_model = gensim.models.LsiModel.load('./static/model_207.lsi')
    return lsa_model


if __name__ == '__main__':
    app.gamesage_database = load_gamesage_database()
    app.term_id_dictionary = load_term_id_dictionary()
    app.tf_idf_model = load_tf_idf_model()
    app.lsa_model = load_lsa_model()
    app.run(debug=False)
else:
    app.gamesage_database = load_gamesage_database()
    app.term_id_dictionary = load_term_id_dictionary()
    app.tf_idf_model = load_tf_idf_model()
    app.lsa_model = load_lsa_model()
if not app.debug:
    import logging
    file_handler = logging.FileHandler('gamesage.log')
    file_handler.setLevel(logging.WARNING)
    app.logger.addHandler(file_handler)
