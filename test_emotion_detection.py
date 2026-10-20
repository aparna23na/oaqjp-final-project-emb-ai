"""Unit tests for the emotion_detector function."""
import unittest

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Check the dominant emotion detected for a range of statements."""

    def test_emotion_detector(self):
        """Each statement must map to the expected dominant emotion."""
        cases = [
            ("I am glad this happened", "joy"),
            ("I am really mad about this", "anger"),
            ("I feel disgusted just hearing about this", "disgust"),
            ("I am so sad about this", "sadness"),
            ("I am really afraid that this will happen", "fear"),
        ]
        for statement, expected in cases:
            with self.subTest(statement=statement):
                result = emotion_detector(statement)
                self.assertEqual(result["dominant_emotion"], expected)


if __name__ == "__main__":
    unittest.main()
