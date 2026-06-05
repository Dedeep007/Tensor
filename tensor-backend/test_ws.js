
const WebSocket = require('ws');
const ws = new WebSocket('ws://localhost:3000');

ws.on('open', function open() {
  console.log('Connected to backend');
  ws.send(JSON.stringify({
    prompt: 'Hello, what are you?',
    activeFiles: [],
    llmConfig: { provider: 'openai', apiKey: 'MOCK_KEY', modelName: 'gpt-4o' }
  }));
});

ws.on('message', function incoming(data) {
  console.log('Received: ' + data);
});

ws.on('error', console.error);
