from flask import Flask, Response, render_template, request, session
from flask_socketio import SocketIO, send, emit, join_room, leave_room
import os
import requests
import base64
import time
from scipy.io.wavfile import write as write_wav
from infer import restore_model, load_audio
# from g2pNp2g_simple.p2gFuntion import p2g_simple as p2g

app = Flask(__name__)
app.config["SECRET_KEY"] = "dangvansam"
socketio = SocketIO(app)

config = 'model_english/quartznet15x5.yaml'
encoder_checkpoint = 'model_english/JasperEncoder-STEP-247400.pt'
decoder_checkpoint = 'model_english/JasperDecoderForCTC-STEP-247400.pt'

neural_factory = restore_model(config, encoder_checkpoint, decoder_checkpoint, lm=False)
print('restore model checkpoint done!')

@app.route("/")
def index():
    return render_template("socketio_template/index.html", audio_path=None, async_mode=socketio.async_mode)

@socketio.on('connect')
def connected():
    print("CONNECTED: " + request.sid)
    emit('to_client', {'text':request.sid})

@socketio.on('to_server')
def response_to_client(data):
    print(data["text"])
    emit('to_client',{'text':len(data["text"].split())})
    
@socketio.on('audio_to_server')
def get_audio(data):
    #print(data)
    filename = time.strftime("%Y%m%d_%H%M%S")
    filepath = "static/record/" + filename + ".wav" 
    audio_file = open(filepath, "wb")
    decode_string = base64.b64decode(data["audio_base64"].split(",")[1])
    audio_file.write(decode_string)
    #asr
    print("asr processing...")
    sig = load_audio(filepath)
    greedy_hypotheses = neural_factory.infer_signal(sig)
    print('greedy predict:{}'.format(greedy_hypotheses))
    print("asr completed")
    emit('audio_to_client', {'text_beamlm': "not use LM", 'text_greedy':greedy_hypotheses, 'filepath':filepath})

@socketio.on('image-upload')
def imageUpload(image):
    emit('send-image', image, broadcast = True)

@app.route('/upload', methods=['POST'])
def predic_upload():
    print('upload file')
    if request.method == 'POST':
        _file = request.files['file']
        if _file.filename == '':
            return index()
        print('\n\nfile uploaded:',_file.filename)
        _file.save('static/upload/' + _file.filename)
        print('Write file success!')
        sig = load_audio('static/upload/'+ _file.filename)
        greedy_hypotheses = neural_factory.infer_signal(sig)
        print('greedy predict:{}'.format(greedy_hypotheses))
        # print('beamLM predict:{}'.format(beam_hypotheses))

        return render_template('socketio_template/index.html', greedy_predict=greedy_hypotheses, beam_predict="not use LM", audio_path='static/upload/' + _file.filename)

if __name__ == '__main__':
    socketio.run(app, host="localhost", port=5001, ssl_context="adhoc", debug=False)
