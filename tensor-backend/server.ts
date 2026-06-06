import express from 'express';
import { WebSocketServer, WebSocket } from 'ws';
import http from 'http';
import { graph, getLLM } from './graph';
import { activeProcesses, diffEvents } from './tools';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import * as fs from 'fs/promises';
import * as path from 'path';

const app = express();
app.use(express.json());
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

async function findFileRecursively(dir: string, fileName: string): Promise<string | null> {
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name === 'node_modules' || entry.name === '.git' || entry.name === 'dist') continue;
      
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        const found = await findFileRecursively(fullPath, fileName);
        if (found) return found;
      } else if (entry.name === fileName) {
        return fullPath;
      }
    }
  } catch (e) {
    // ignore
  }
  return null;
}

async function resolveMentions(prompt: string, workspaceRoot: string): Promise<string> {
  if (!workspaceRoot) return prompt;
  
  const mentionRegex = /@([a-zA-Z0-9_\-\.]+)/g;
  let match;
  let resolvedContext = "";
  let finalPrompt = prompt;

  while ((match = mentionRegex.exec(prompt)) !== null) {
    const fileName = match[1];
    const foundPath = await findFileRecursively(workspaceRoot, fileName);
    if (foundPath) {
      try {
        const content = await fs.readFile(foundPath, 'utf8');
        resolvedContext += `\n--- Context from @${fileName} (${foundPath}) ---\n${content}\n`;
      } catch (e) {}
    }
  }

  if (resolvedContext.length > 0) {
    finalPrompt = `[USER PROVIDED EXPLICIT FILE MENTIONS:]\n${resolvedContext}\n\n[USER INSTRUCTION:]\n${prompt}`;
  }
  return finalPrompt;
}

async function generateWorkspaceMap(dir: string, prefix = ""): Promise<string> {
  let tree = "";
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      if (entry.name === 'node_modules' || entry.name === '.git' || entry.name === 'dist') continue;
      
      const isLast = i === entries.length - 1;
      tree += `${prefix}${isLast ? "└── " : "├── "}${entry.name}\n`;
      if (entry.isDirectory()) {
        tree += await generateWorkspaceMap(path.join(dir, entry.name), prefix + (isLast ? "    " : "│   "));
      }
    }
  } catch (e) {}
  return tree;
}

wss.on('connection', (ws: WebSocket) => {
  console.log('Client connected to Tensor IDE Backend');
  let abortController: AbortController | null = null;

  ws.on('message', async (message: string) => {
    try {
      const data = JSON.parse(message);
      const { prompt, activeFiles, activeFileContent, activeFilePath, llmConfig, workspaceRoot, threadId } = data;

      if (data.type === 'clearHistory') {
         // UI manages thread IDs locally, so backend doesn't strictly need to do anything, 
         // but we can acknowledge it.
         ws.send(JSON.stringify({ type: 'system', data: `Chat history cleared (New Thread).` }));
         return;
      }

      if (data.type === 'stop') {
        if (abortController) {
          abortController.abort("User cancelled the stream.");
          abortController = null;
        }
        return;
      }

      if (data.type === 'cancelTask') {
        if (data.taskId) {
          const child = activeProcesses.get(data.taskId);
          if (child) {
            child.kill();
            activeProcesses.delete(data.taskId);
            ws.send(JSON.stringify({ type: 'system', data: `Task ${data.taskId} cancelled by UI.` }));
          }
        } else {
          let count = 0;
          for (const [id, child] of activeProcesses.entries()) {
            child.kill();
            activeProcesses.delete(id);
            count++;
          }
          if (count > 0) {
            ws.send(JSON.stringify({ type: 'system', data: `Cancelled ${count} active background task(s).` }));
          }
        }
        return;
      }

      if (data.type === 'diffResponse') {
        diffEvents.emit(`response_${data.id}`, data.accepted);
        return;
      }

      if (data.type === 'inlineEdit') {
        const { prompt, code, file, startLine, endLine, llmConfig } = data;
        const llm = getLLM(llmConfig);
        const sysMsg = new SystemMessage("You are an expert coding assistant. You have been asked to edit a specific block of code. Output ONLY the raw, modified code. DO NOT wrap it in markdown blockticks (```). DO NOT explain your changes. Just output the raw code snippet that will replace the user's selection.");
        const usrMsg = new HumanMessage(`File: ${file}\nLines: ${startLine}-${endLine}\n\nOriginal Code:\n${code}\n\nInstruction:\n${prompt}`);
        
        try {
          const res = await llm.invoke([sysMsg, usrMsg]);
          let newCode = typeof res.content === "string" ? res.content : res.content.toString();
          
          // Fallback strip markdown if model hallucinates it
          if (newCode.startsWith("```")) {
             newCode = newCode.replace(/```[a-z]*\n/g, "").replace(/```$/g, "");
          }

          ws.send(JSON.stringify({
            type: 'inlineEditComplete',
            file,
            startLine,
            endLine,
            newCode: newCode.trimEnd()
          }));
        } catch (e: any) {
          ws.send(JSON.stringify({ type: 'error', data: `Inline edit failed: ${e.message}` }));
        }
        return;
      }

      // Resolve @ mentions
      let resolvedPrompt = await resolveMentions(prompt, workspaceRoot || process.cwd());

      // Parse /map
      if (resolvedPrompt.startsWith('/map')) {
        const workspaceMap = await generateWorkspaceMap(workspaceRoot || process.cwd());
        resolvedPrompt = `[WORKSPACE ARCHITECTURE MAP]\n${workspaceMap}\n\n` + resolvedPrompt.replace('/map', '').trim();
      }

      // Inject active file context seamlessly
      if (activeFileContent && activeFilePath) {
        resolvedPrompt = `[USER'S CURRENTLY ACTIVE EDITOR TAB: ${activeFilePath}]\n\`\`\`\n${activeFileContent}\n\`\`\`\n\n` + resolvedPrompt;
      }

      if (abortController) {
        abortController.abort("New stream started.");
      }
      abortController = new AbortController();

      const stream = await graph.streamEvents({
        messages: [new HumanMessage(resolvedPrompt)],
        activeFiles: activeFiles || [],
        workspaceRoot: workspaceRoot || "",
        llmConfig: llmConfig || { provider: "groq", apiKey: "", modelName: "llama3-70b-8192" },
      }, { version: "v2", signal: abortController.signal, configurable: { thread_id: threadId || "default" } });

      for await (const event of stream) {
        if (event.tags && event.tags.includes("hide")) continue;

        if (event.event === "on_chat_model_stream") {
            ws.send(JSON.stringify({ type: 'token', data: event.data.chunk.content }));
        } else if (event.event === "on_tool_start") {
            ws.send(JSON.stringify({ type: 'tool_start', data: event.name, input: event.data.input }));
        } else if (event.event === "on_tool_end") {
            ws.send(JSON.stringify({ type: 'tool_end', data: event.name, output: event.data.output }));
        } else if (event.event === "on_chain_end" && event.name === "LangGraph") {
            ws.send(JSON.stringify({ type: 'state_update', data: event.data.output }));
        }
      }
      ws.send(JSON.stringify({ type: 'done' }));

    } catch (error: any) {
      ws.send(JSON.stringify({ type: 'error', data: error.message }));
    }
  });

  // Forward diff proposals to this connection
  const onPropose = (data: any) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'proposeDiff', ...data }));
    }
  };
  
  const onEditComplete = (data: any) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'editComplete', ...data }));
    }
  };

  diffEvents.on('propose', onPropose);
  diffEvents.on('editComplete', onEditComplete);

  ws.on('close', () => {
    diffEvents.off('propose', onPropose);
    diffEvents.off('editComplete', onEditComplete);
    console.log('Client disconnected');
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Tensor IDE Backend running on http://localhost:${PORT}`);
});
