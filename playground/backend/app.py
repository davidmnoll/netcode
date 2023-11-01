from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message event')
def handle_message(data):
    print('Received message:', data)
    emit('response event', {'data': 'Server received your message!'}, broadcast=True)

if __name__ == '__main__':
    socketio.run(app)