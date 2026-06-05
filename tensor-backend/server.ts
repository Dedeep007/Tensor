import express from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import http from 'http';
import { graph } from './graph';
import { HumanMessage } from '@langchain/core/messages';

const app = express();
app.use(express.json());
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

wss.on('connection', (ws: WebSocket) => {
  console.log('Client connected to Tensor IDE Backend');

  ws.on('message', async (message: string) => {
    try {
      const data = JSON.parse(message);
      const { prompt, activeFiles, llmConfig } = data;

      const stream = await graph.streamEvents({
        messages: [new HumanMessage(prompt)],
        activeFiles: activeFiles || [],
        llmConfig: llmConfig || { provider: "groq", apiKey: "", modelName: "llama3-70b-8192" },
      }, { version: "v2" });

      for await (const event of stream) {
        if (event.event === "on_chat_model_stream") {
            ws.send(JSON.stringify({ type: 'token', data: event.data.chunk.content }));
        } else if (event.event === "on_tool_start") {
            ws.send(JSON.stringify({ type: 'tool_start', data: event.name }));
        } else if (event.event === "on_tool_end") {
            ws.send(JSON.stringify({ type: 'tool_end', data: event.name }));
        } else if (event.event === "on_chain_end" && event.name === "LangGraph") {
            ws.send(JSON.stringify({ type: 'state_update', data: event.data.output }));
        }
      }
      ws.send(JSON.stringify({ type: 'done' }));

    } catch (error: any) {
      ws.send(JSON.stringify({ type: 'error', data: error.message }));
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Tensor IDE Backend running on http://localhost:${PORT}`);
});
