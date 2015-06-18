from fileman.file_getter import get_file

__author__ = 'peter'


def read_the_bible(max_characters = None):
    """
    Returns the King James Bible as a single string.
    Thanks to Janel (http://janelwashere.com/pages/bible_daily_reading.html) for compiling it.
    :param max_characters: You have the option to truncate it to a length of max_characters
        (If you're Jewish, for instance)
    :return: A string.
    """

    filename = get_file(
        relative_name = 'data/king_james_bible.txt',
        url = 'http://janelwashere.com/files/bible_daily.txt',
        )

    with open(filename) as f:
        text = f.read(-1 if max_characters is None else max_characters)

    return text
