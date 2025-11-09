from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import urlparse, parse_qs
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

app = Flask(__name__)
CORS(app)   # <-- Only initialize once

API_KEY = "AIzaSyCRhWpyh_AOOBZecFZ5iJlTBe8DX76XzuU"   # <--- Put your YouTube API key here


def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    elif query.hostname in ('www.youtube.com', 'youtube.com'):
        return parse_qs(query.query)['v'][0]
    return None


def fetch_replies(parent_thread_id, api_key):
    replies = []
    base = "https://www.googleapis.com/youtube/v3/comments"
    params = {
        "part": "snippet",
        "parentId": parent_thread_id,
        "key": api_key,
        "maxResults": 100,
        "textFormat": "plainText"
    }

    nextToken = None
    while True:
        if nextToken:
            params["pageToken"] = nextToken

        r = requests.get(base, params=params)
        d = r.json()

        for it in d.get("items", []):
            replies.append(it["snippet"]["textDisplay"])

        nextToken = d.get("nextPageToken")
        if not nextToken:
            break

    return replies


def get_comments(video_id, api_key, include_replies=True, max_comments=5000):
    comments = []
    base = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": api_key,
        "maxResults": 100,
        "textFormat": "plainText"
    }

    nextPageToken = None
    while True:
        if nextPageToken:
            params["pageToken"] = nextPageToken

        resp = requests.get(base, params=params)
        data = resp.json()

        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(top)

            if include_replies and item["snippet"].get("totalReplyCount", 0) > 0:
                replies = fetch_replies(item["id"], api_key)
                comments.extend(replies)

            if len(comments) >= max_comments:
                return comments[:max_comments]

        nextPageToken = data.get("nextPageToken")
        if not nextPageToken:
            break

        time.sleep(0.1)

    return comments


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    video_url = data.get("url", "")
    video_id = extract_video_id(video_url)

    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    comments = get_comments(video_id, API_KEY, include_replies=True, max_comments=5000)

    if not comments:
        return jsonify({"message": "No comments found"}), 404

    sid = SentimentIntensityAnalyzer()
    positive = negative = neutral = 0

    for comment in comments:
        comp = sid.polarity_scores(comment)["compound"]
        if comp >= 0.2:
            positive += 1
        elif comp <= -0.2:
            negative += 1
        else:
            neutral += 1

    return jsonify({
        "total_comments": len(comments),
        "positive": positive,
        "negative": negative,
        "neutral": neutral
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
