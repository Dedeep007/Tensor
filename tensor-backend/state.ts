import { BaseMessage } from "@langchain/core/messages";
import { Annotation } from "@langchain/langgraph";

export const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  activeFiles: Annotation<string[]>({
    reducer: (x, y) => [...new Set([...x, ...y])],
    default: () => [],
  }),
  sessionContext: Annotation<string>({
    reducer: (x, y) => y,
    default: () => "",
  }),
  llmConfig: Annotation<any>({
    reducer: (x, y) => y,
    default: () => ({ provider: "groq", apiKey: "", modelName: "llama3-70b-8192" }),
  }),
  fastRecallPointers: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  longHorizonKnowledge: Annotation<any[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  activeAgent: Annotation<string>({
    reducer: (x, y) => y,
    default: () => "supervisor",
  }),
  subagentTasks: Annotation<any[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
});

export type State = typeof GraphState.State;
