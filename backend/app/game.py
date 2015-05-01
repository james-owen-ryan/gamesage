class Game(object):
    """A game representation for the purposes of GameSage."""

    def __init__(self, game_id, title, lsa_vector_str):
        """Initialize a Game object."""
        self.id = game_id
        self.title = title.decode('utf-8')
        self.lsa_vector = self.parse_lsa_vector_str(lsa_vector_str)
        self.gamenet_link = self.get_link_to_gamenet_entry(game_id, title)

    @staticmethod
    def parse_lsa_vector_str(lsa_vector_str):
        """Parse a string specifying an LSA vector to return a list representation of it."""
        lsa_vector = [float(i) for i in lsa_vector_str.split(',')[1:]]  # Exclude first dimension
        # Add in the dimension indices -- these are needed for folding in
        lsa_vector_with_indices = []
        for i in xrange(len(lsa_vector)):
            index_of_this_dimension = i+1
            value_along_this_dimension = lsa_vector[i]
            lsa_vector_with_indices.append((index_of_this_dimension, value_along_this_dimension))
        return lsa_vector_with_indices

    @staticmethod
    def get_link_to_gamenet_entry(game_id, title):
        """Return a link to this game's Gamenet entry."""
        url = "http://gamecip-projects.soe.ucsc.edu/gamenet/games/"
        url += game_id
        link = "<a href={} target=_blank>".format(url) + title + "</a>"
        return link