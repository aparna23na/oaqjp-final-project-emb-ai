"""Emotion detection using the Watson NLP EmotionPredict service."""
import json

import requests

URL = ("https://sn-watson-emotion.labs.skills.network/v1/"
       "watson.runtime.nlp.v1/NlpService/EmotionPredict")
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_predictor(response_text):
    """Format the raw Watson response text into scores plus the dominant emotion."""
    scores = json.loads(response_text)["emotionPredictions"][0]["emotion"]
    result = {emotion: scores[emotion] for emotion in EMOTIONS}
    result["dominant_emotion"] = max(EMOTIONS, key=lambda emotion: result[emotion])
    return result


def emotion_detector(text_to_analyze):
    """Analyze the emotions in the given text.

    Returns a dictionary with the score of each emotion and the dominant
    emotion. If the service rejects the input (status code 400, e.g. blank
    text), every value in the dictionary is None.
    """
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(URL, json=payload, headers=HEADERS, timeout=30)

    if response.status_code == 400:
        return {emotion: None for emotion in EMOTIONS + ("dominant_emotion",)}

    response.raise_for_status()
    return emotion_predictor(response.text)
