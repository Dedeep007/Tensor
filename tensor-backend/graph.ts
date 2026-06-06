import { StateGraph, START, END, MemorySaver } from "@langchain/langgraph";
const memory = new MemorySaver();
import { ChatGroq } from "@langchain/groq";
import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatMistralAI } from "@langchain/mistralai";
import { ChatCohere } from "@langchain/cohere";
import { ChatOllama } from "@langchain/ollama";
import { ChatBedrockConverse } from "@langchain/aws";
import { GraphState, State } from "./state";
import { 
  readFile, 
  createFile,
  preciseCodeEdit, 
  readFileLines,
  multiLineEdit, 
  findFiles, 
  findTextInFiles, 
  listDirectory, 
  readProjectStructure, 
  getErrors, 
  applyPatch,
  runBackgroundCommand,
  cancelCommand,
  searchCodebase
} from "./tools";
import * as fs from "fs/promises";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

export function getLLM(config: any): any {
  if (config.provider === "openai") {
    return new ChatOpenAI({ modelName: config.modelName || "gpt-4o", openAIApiKey: config.apiKey, temperature: 0 });
  } else if (config.provider === "anthropic") {
    return new ChatAnthropic({ modelName: config.modelName || "claude-3-5-sonnet-20240620", anthropicApiKey: config.apiKey, temperature: 0 });
  } else if (config.provider === "google") {
    return new ChatGoogleGenerativeAI({ model: config.modelName || "gemini-1.5-pro", apiKey: config.apiKey, temperature: 0 });
  } else if (config.provider === "mistral") {
    return new ChatMistralAI({ modelName: config.modelName || "mistral-large-latest", apiKey: config.apiKey, temperature: 0 });
  } else if (config.provider === "cohere") {
    return new ChatCohere({ model: config.modelName || "command-r-plus", apiKey: config.apiKey, temperature: 0 });
  } else if (config.provider === "ollama") {
    return new ChatOllama({ model: config.modelName || "llama3.1", temperature: 0, baseUrl: config.baseUrl });
  } else if (config.provider === "aws-bedrock") {
    return new ChatBedrockConverse({ model: config.modelName || "anthropic.claude-3-sonnet-20240229-v1:0", temperature: 0, region: config.region, credentials: config.credentials });
  } else {
    // Default to Groq
    return new ChatGroq({ model: config.modelName || "llama3-70b-8192", apiKey: config.apiKey, temperature: 0 });
  }
}

async function Viewer_Node(state: State): Promise<Partial<State>> {
  let context = "";
  
  if (state.workspaceRoot) {
    try {
      // Add top-level workspace awareness
      const entries = await fs.readdir(state.workspaceRoot, { withFileTypes: true });
      context += `\n--- Workspace Root (${state.workspaceRoot}) ---\n`;
      context += entries.map(e => `${e.isDirectory() ? "[DIR]" : "[FILE]"} ${e.name}`).join("\n") + "\n";
    } catch (e) {
      context += `\n--- Workspace Root ---\nError reading directory.\n`;
    }

    try {
      // Check for .xsorrules
      const rulesPath = `${state.workspaceRoot}/.xsorrules`;
      const rulesContent = await fs.readFile(rulesPath, "utf8");
      context += `\n--- .xsorrules (Project Instructions) ---\n${rulesContent}\n`;
    } catch (e) {
      // No .xsorrules found, ignore
    }
  }

  for (const file of state.activeFiles) {
    try {
      const content = await readFile.invoke({ filePath: file });
      context += `\n--- File: ${file} ---\n${content}\n`;
    } catch (e) {
      context += `\n--- File: ${file} ---\nError reading file.\n`;
    }
  }
  return { sessionContext: context };
}

async function Compactor_Node(state: State): Promise<Partial<State>> {
  const llm = getLLM(state.llmConfig);
  const MAX_CHARS = 4000;
  if (state.sessionContext.length > MAX_CHARS) {
    const summaryPrompt = `Summarize the following code context briefly to save tokens: \n\n${state.sessionContext}`;
    const response = await llm.invoke([new HumanMessage(summaryPrompt)], { tags: ["hide"] });
    return {
      sessionContext: "",
      fastRecallPointers: [response.content as string],
    };
  }
  return {};
}

// Supervisor dispatches tasks to subagents
async function Supervisor_Node(state: State): Promise<Partial<State>> {
  const llm = getLLM(state.llmConfig);
  // Define tools representing the sub-agents for the LLM to call
  const subagentTools = [
    {
      name: "route_to_search",
      description: "Dispatch the search sub-agent to explore the codebase, find files, or read project structure.",
      schema: { type: "object", properties: { query: { type: "string" } } }
    },
    {
      name: "route_to_editor",
      description: "Dispatch the editor sub-agent to modify code, apply patches, or write new files.",
      schema: { type: "object", properties: { instructions: { type: "string" } } }
    },
    {
      name: "route_to_execution",
      description: "Dispatch the execution sub-agent to run tests, fetch compilation errors, or validate builds.",
      schema: { type: "object", properties: { command: { type: "string" } } }
    }
  ];
  
  const supervisorLlm = llm.bindTools(subagentTools);
  const messages = [
    new SystemMessage(`You are the Supervisor node in an agentic IDE. Your job is to orchestrate tasks by routing to the appropriate sub-agents. 
CRITICAL RULE: If the user asks to create, write, modify, or edit code/files, you MUST route to the 'Editor_Subagent' using the 'route_to_editor' tool. Do NOT just output the code in your response. The Editor sub-agent has the tools to actually create and edit files on disk.
If the user's request is just a question and is fulfilled, respond to the user directly without routing.
Context: ${state.sessionContext} \nPointers: ${state.fastRecallPointers.join(" | ")}`),
    ...state.messages,
  ];
  
  const response = await supervisorLlm.invoke(messages);
  
  // Extract intent
  let activeAgent = "supervisor";
  if (response.tool_calls && response.tool_calls.length > 0) {
    const toolName = response.tool_calls[0].name;
    if (toolName === "route_to_search") activeAgent = "search";
    if (toolName === "route_to_editor") activeAgent = "editor";
    if (toolName === "route_to_execution") activeAgent = "execution";
  }
  
  return { messages: [response], activeAgent };
}

async function Search_Subagent(state: State): Promise<Partial<State>> {
  const llm = getLLM(state.llmConfig);
  const searchLlm = llm.bindTools([listDirectory, searchCodebase, readFile, readFileLines]);
  const messages = [
    new SystemMessage("You are the Search Sub-agent. Explore directories and search the codebase for relevant functions or definitions."),
    ...state.messages,
  ];
  const response = await searchLlm.invoke(messages);
  return { messages: [response], activeAgent: "supervisor" }; // return control to supervisor
}

async function Editor_Subagent(state: State): Promise<Partial<State>> {
  const llm = getLLM(state.llmConfig);
  const editorLlm = llm.bindTools([preciseCodeEdit, multiLineEdit, applyPatch, createFile, readFile, readFileLines]);
  const messages = [
    new SystemMessage("You are the Editor Sub-agent. Execute edits safely without hallucination."),
    ...state.messages,
  ];
  const response = await editorLlm.invoke(messages);
  return { messages: [response], activeAgent: "supervisor" };
}

async function Execution_Subagent(state: State): Promise<Partial<State>> {
  const llm = getLLM(state.llmConfig);
  const execLlm = llm.bindTools([getErrors, runBackgroundCommand, cancelCommand]);
  const messages = [
    new SystemMessage("You are the Execution Sub-agent. Fetch compilation errors or manage background processes."),
    ...state.messages,
  ];
  const response = await execLlm.invoke(messages);
  return { messages: [response], activeAgent: "supervisor" };
}

function routeFromSupervisor(state: State) {
  if (state.activeAgent === "search") return "Search_Subagent";
  if (state.activeAgent === "editor") return "Editor_Subagent";
  if (state.activeAgent === "execution") return "Execution_Subagent";
  return END;
}

export const graph = new StateGraph(GraphState)
  .addNode("Viewer_Node", Viewer_Node)
  .addNode("Compactor_Node", Compactor_Node)
  .addNode("Supervisor_Node", Supervisor_Node)
  .addNode("Search_Subagent", Search_Subagent)
  .addNode("Editor_Subagent", Editor_Subagent)
  .addNode("Execution_Subagent", Execution_Subagent)
  .addEdge(START, "Viewer_Node")
  .addEdge("Viewer_Node", "Compactor_Node")
  .addEdge("Compactor_Node", "Supervisor_Node")
  .addConditionalEdges("Supervisor_Node", routeFromSupervisor, {
    Search_Subagent: "Search_Subagent",
    Editor_Subagent: "Editor_Subagent",
    Execution_Subagent: "Execution_Subagent",
    [END]: END,
  })
  .addEdge("Search_Subagent", "Supervisor_Node")
  .addEdge("Editor_Subagent", "Supervisor_Node")
  .addEdge("Execution_Subagent", "Supervisor_Node")
  .compile({ checkpointer: memory });
