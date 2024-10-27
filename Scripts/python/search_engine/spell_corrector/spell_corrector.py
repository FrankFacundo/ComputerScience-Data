from nltk.corpus import words
from rapidfuzz import fuzz, process


class FastLevenshteinSpellCorrector:
    def __init__(self):
        # Create a set of words to avoid duplicates and use it for fast lookup
        self.word_list = list(set(words.words()))

    def correct(self, word, n=3):
        """Suggests corrections based on Levenshtein distance with optimized search."""
        suggestions = process.extract(word, self.word_list, limit=n, scorer=fuzz.ratio)
        return [
            suggestion[0] for suggestion in suggestions if suggestion[1] >= 80
        ]  # Only include high similarity matches


# Instantiate and test the new class with examples
fast_levenshtein_corrector = FastLevenshteinSpellCorrector()

# Test example
misspelled_word = "moother"
print("Fast Levenshtein Spell Correction Suggestions:")
print(fast_levenshtein_corrector.correct(misspelled_word))
