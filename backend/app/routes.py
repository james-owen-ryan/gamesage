import csv
from flask import Flask, render_template, jsonify
from gamesage import GameSage
from game import Game


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submittedText=<user_submitted_text>')
def generate_gamenet_query(user_submitted_text):
    gamesage = GameSage(database=app.database, user_submitted_text=user_submitted_text)
    print user_submitted_text
    return jsonify(
        user_submitted_text=user_submitted_text,
        most_related_games_str=gamesage.most_related_games_str,
        least_related_games_str=gamesage.least_related_games_str
    )


def load_database():
    """Load the database of game representations from a TSV file."""
    database = []
    with open('static/game_lsa_vectors.tsv', 'r') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')
        for row in reader:
            game_id, title, year, lsa_vector_str = row
            game_object = Game(game_id, title, lsa_vector_str)
            database.append(game_object)
    return database


if __name__ == '__main__':
    app.database = load_database()
    app.run(debug=False)
else:
    app.database = load_database()
if not app.debug:
    import logging
    file_handler = logging.FileHandler('gamesage.log')
    file_handler.setLevel(logging.WARNING)
    app.logger.addHandler(file_handler)
