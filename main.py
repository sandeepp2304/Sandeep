from flask import Flask
from flask_restplus import Api, Resource, fields, reqparse
from modules.audio_sentiment_analysis import AudioSentimentAnalysis
from raven.contrib.flask import Sentry


app = Flask(__name__)

app.config["RESTPLUS_MASK_SWAGGER"] = False

api = Api(app, version='1.0', title='Audio Sentiment Analysis Service')

audio_sentiment_service = api.namespace('', description='Audio_Sentiment_service endpoints')

path = api.namespace('audio_path', description='Audio Path endpoints')

sentry = Sentry(app)

audio_path = reqparse.RequestParser()
audio_path.add_argument('path', type=str, help='audio path', required=True)


@path.route('/audio_sentiment_analysis')
@path.response(500, "{'message': 'file not found'}")
@path.response(200, '{"message": "processing"}')
class AudioAnalysis(Resource):
    @path.doc('audio path')
    @path.expect(audio_path, validate=True)
    # @search.marshal_with(data)
    def get(self):
        args = audio_path.parse_args()
        obj = AudioSentimentAnalysis()
        response = obj.main(args['path'])

        return response


if __name__ == '__main__':
    app.run(debug=True, port=5005)
