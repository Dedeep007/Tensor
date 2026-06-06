import { tool } from "@langchain/core/tools";
import { z } from "zod";
import * as fs from "fs/promises";
import * as path from "path";
import { exec, spawn, ChildProcess } from "child_process";
import { promisify } from "util";
import { EventEmitter } from "events";

const execAsync = promisify(exec);
export const diffEvents = new EventEmitter();

export const activeProcesses = new Map<string, ChildProcess>();

export const runBackgroundCommand = tool(
  async ({ command, cwd }) => {
    try {
      const taskId = Math.random().toString(36).substring(7);
      const child = spawn(command, { shell: true, cwd: cwd ? path.resolve(cwd) : process.cwd() });
      
      activeProcesses.set(taskId, child);
      
      child.on('exit', () => {
        activeProcesses.delete(taskId);
      });

      return `Started background command: '${command}'. Task ID: ${taskId}. You can use this ID to cancel it.`;
    } catch (e: any) {
      return `Error starting command: ${e.message}`;
    }
  },
  {
    name: "run_background_command",
    description: "Runs a long-running terminal command (e.g. npm start, server) in the background. Does not wait for output.",
    schema: z.object({
      command: z.string().describe("The shell command to run"),
      cwd: z.string().optional().describe("Optional working directory")
    })
  }
);

export const cancelCommand = tool(
  async ({ taskId }) => {
    const child = activeProcesses.get(taskId);
    if (!child) return `Task ${taskId} not found or already finished.`;
    
    child.kill();
    activeProcesses.delete(taskId);
    return `Task ${taskId} cancelled successfully.`;
  },
  {
    name: "cancel_command",
    description: "Cancels a running background command.",
    schema: z.object({
      taskId: z.string().describe("The Task ID returned by run_background_command")
    })
  }
);

async function searchDirectory(dir: string, query: string, extensions: string[], results: string[] = []): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name === "node_modules" || entry.name === ".git" || entry.name === "dist") continue;
    
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await searchDirectory(fullPath, query, extensions, results);
    } else {
      if (extensions.length > 0 && !extensions.some(ext => entry.name.endsWith(ext))) continue;
      
      try {
        const content = await fs.readFile(fullPath, "utf-8");
        const lines = content.split('\n');
        for (let i = 0; i < lines.length; i++) {
          if (lines[i].includes(query)) {
            results.push(`${fullPath}:${i + 1}: ${lines[i].trim()}`);
            if (results.length > 50) return results; // Cap results
          }
        }
      } catch (e) {
        // Ignore binary or unreadable files
      }
    }
  }
  return results;
}

export const searchCodebase = tool(
  async ({ query, directory, fileExtensions }) => {
    try {
      const exts = fileExtensions ? fileExtensions.split(',').map(e => e.trim()) : [];
      const dirToSearch = directory ? path.resolve(directory) : process.cwd();
      const results = await searchDirectory(dirToSearch, query, exts);
      
      if (results.length === 0) return `No matches found for '${query}'.`;
      return `Found matches:\n${results.join('\n')}`;
    } catch (e: any) {
      return `Error searching codebase: ${e.message}`;
    }
  },
  {
    name: "search_codebase",
    description: "Searches the entire codebase for a string or variable. Extremely useful for finding where functions or classes are defined.",
    schema: z.object({
      query: z.string().describe("The exact text or code snippet to search for"),
      directory: z.string().optional().describe("The absolute path of the workspace directory to search in"),
      fileExtensions: z.string().optional().describe("Comma-separated extensions to filter by (e.g. '.ts,.js,.html'). Empty means all.")
    })
  }
);

export const readFile = tool(
  async ({ filePath }) => {
    try {
      const content = await fs.readFile(path.resolve(filePath), "utf8");
      return content;
    } catch (e: any) {
      return `Error reading file: ${e.message}`;
    }
  },
  {
    name: "read_file",
    description: "Reads the content of a file from the local filesystem.",
    schema: z.object({
      filePath: z.string().describe("The absolute or relative path to the file to read"),
    }),
  }
);

export const createFile = tool(
  async ({ filePath, content }) => {
    try {
      const absPath = path.resolve(filePath);
      
      return new Promise((resolve) => {
        const diffId = Math.random().toString(36).substring(7);
        diffEvents.emit('propose', { id: diffId, file: absPath, oldContent: "", newContent: content, type: 'create' });
        
        diffEvents.once(`response_${diffId}`, async (accepted: boolean) => {
          if (accepted) {
            await fs.writeFile(absPath, content, "utf8");
            const newLines = content.split('\n').length;
            diffEvents.emit('editComplete', { file: path.basename(absPath), added: newLines, removed: 0 });
            resolve(`Successfully created file: ${filePath}`);
          } else {
            resolve(`User REJECTED the creation of file: ${filePath}`);
          }
        });
      });
    } catch (e: any) {
      return `Error creating file: ${e.message}`;
    }
  },
  {
    name: "create_file",
    description: "Creates a new file at the specified path with the provided content.",
    schema: z.object({
      filePath: z.string().describe("The absolute or relative path where the new file should be created"),
      content: z.string().describe("The complete string content to write into the new file"),
    }),
  }
);

export const readFileLines = tool(
  async ({ filePath, startLine, endLine }) => {
    try {
      const content = await fs.readFile(path.resolve(filePath), "utf8");
      const lines = content.split('\n');
      // 1-indexed to 0-indexed
      const targetLines = lines.slice(Math.max(0, startLine - 1), Math.min(lines.length, endLine));
      
      let result = `--- ${filePath} (Lines ${startLine}-${endLine}) ---\n`;
      targetLines.forEach((line, idx) => {
        result += `${startLine + idx}: ${line}\n`;
      });
      return result;
    } catch (e: any) {
      return `Error reading file lines: ${e.message}`;
    }
  },
  {
    name: "read_file_lines",
    description: "Reads a specific range of lines from a file, returning them with line numbers. Use this to inspect large files without blowing up context window.",
    schema: z.object({
      filePath: z.string().describe("The path to the file"),
      startLine: z.number().describe("The starting line number (1-indexed)"),
      endLine: z.number().describe("The ending line number (inclusive)")
    }),
  }
);

export const multiLineEdit = tool(
  async ({ filePath, startLine, endLine, searchBlock, replaceBlock }) => {
    try {
      const absPath = path.resolve(filePath);
      const content = await fs.readFile(absPath, "utf8");
      const lines = content.split('\n');
      
      const beforeLines = lines.slice(0, Math.max(0, startLine - 1));
      const targetContent = lines.slice(Math.max(0, startLine - 1), Math.min(lines.length, endLine)).join('\n');
      const afterLines = lines.slice(Math.min(lines.length, endLine));
      
      if (!targetContent.includes(searchBlock)) {
        return `Error: The searchBlock was not found exactly within the specified lines [${startLine}-${endLine}].`;
      }
      
      const replacedTarget = targetContent.replace(searchBlock, replaceBlock);
      const newContent = [...beforeLines, replacedTarget, ...afterLines].join('\n');
      
      return new Promise((resolve) => {
        const diffId = Math.random().toString(36).substring(7);
        diffEvents.emit('propose', { id: diffId, file: absPath, oldContent: content, newContent, type: 'edit' });
        
        diffEvents.once(`response_${diffId}`, async (accepted: boolean) => {
          if (accepted) {
            await fs.writeFile(absPath, newContent, "utf8");
            const oldLines = content.split('\n').length;
            const newLines = newContent.split('\n').length;
            const added = newLines > oldLines ? newLines - oldLines : 0;
            const removed = oldLines > newLines ? oldLines - newLines : 0;
            diffEvents.emit('editComplete', { file: path.basename(absPath), added, removed });
            resolve(`Successfully replaced code block in ${filePath} at lines ${startLine}-${endLine}`);
          } else {
            resolve(`User REJECTED the code edit in ${filePath}`);
          }
        });
      });
    } catch (e: any) {
      return `Error editing file: ${e.message}`;
    }
  },
  {
    name: "multi_line_edit",
    description: "Surgically edits a file by replacing a searchBlock with a replaceBlock ONLY within a specified line range.",
    schema: z.object({
      filePath: z.string().describe("The file to edit"),
      startLine: z.number().describe("The starting line number (1-indexed) containing the block to edit"),
      endLine: z.number().describe("The ending line number (1-indexed, inclusive) containing the block"),
      searchBlock: z.string().describe("The exact existing code block to replace. Must match the file exactly within those lines."),
      replaceBlock: z.string().describe("The new code block to insert in place of searchBlock."),
    }),
  }
);

export const preciseCodeEdit = tool(
  async ({ filePath, searchBlock, replaceBlock }) => {
    try {
      const absPath = path.resolve(filePath);
      const content = await fs.readFile(absPath, "utf8");
      
      if (!content.includes(searchBlock)) {
        return `Error: The searchBlock was not found exactly in the file. To prevent hallucinated edits, the search block must match the existing code precisely.`;
      }
      
      const newContent = content.replace(searchBlock, replaceBlock);
      
      return new Promise((resolve) => {
        const diffId = Math.random().toString(36).substring(7);
        diffEvents.emit('propose', { id: diffId, file: absPath, oldContent: content, newContent, type: 'edit' });
        
        diffEvents.once(`response_${diffId}`, async (accepted: boolean) => {
          if (accepted) {
            await fs.writeFile(absPath, newContent, "utf8");
            const oldLines = content.split('\n').length;
            const newLines = newContent.split('\n').length;
            const added = Math.max(0, newLines - oldLines);
            const removed = Math.max(0, oldLines - newLines);
            diffEvents.emit('editComplete', { file: path.basename(absPath), added, removed });
            resolve(`Successfully replaced code block in ${filePath}`);
          } else {
            resolve(`User REJECTED the code edit in ${filePath}`);
          }
        });
      });
    } catch (e: any) {
      return `Error editing file: ${e.message}`;
    }
  },
  {
    name: "precise_code_edit",
    description: "Surgically edits a file by replacing a specific searchBlock with a replaceBlock.",
    schema: z.object({
      filePath: z.string().describe("The file to edit"),
      searchBlock: z.string().describe("The exact existing code block to replace. Must match the file exactly."),
      replaceBlock: z.string().describe("The new code block to insert in place of searchBlock."),
    }),
  }
);

// --- NEW TOOLS FOR V2 ARCHITECTURE ---

export const findFiles = tool(
  async ({ query, dirPath }) => {
    try {
      const results: string[] = [];
      const searchDir = async (currentPath: string) => {
        const entries = await fs.readdir(currentPath, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(currentPath, entry.name);
          if (entry.isDirectory()) {
            if (entry.name !== "node_modules" && entry.name !== ".git") {
              await searchDir(fullPath);
            }
          } else if (entry.name.includes(query) || fullPath.includes(query)) {
            results.push(fullPath);
          }
        }
      };
      await searchDir(path.resolve(dirPath));
      return results.length ? results.join("\n") : "No matching files found.";
    } catch (e: any) {
      return `Error finding files: ${e.message}`;
    }
  },
  {
    name: "find_files",
    description: "Search for files by name/path query within a directory.",
    schema: z.object({
      query: z.string().describe("The file name or pattern to search for"),
      dirPath: z.string().describe("The root directory to search in"),
    }),
  }
);

export const findTextInFiles = tool(
  async ({ query, dirPath }) => {
    try {
      const results: string[] = [];
      const searchDir = async (currentPath: string) => {
        const entries = await fs.readdir(currentPath, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(currentPath, entry.name);
          if (entry.isDirectory()) {
            if (entry.name !== "node_modules" && entry.name !== ".git") {
              await searchDir(fullPath);
            }
          } else {
            try {
              const content = await fs.readFile(fullPath, "utf8");
              if (content.includes(query)) {
                results.push(`Found in ${fullPath}`);
              }
            } catch (err) {
              // Ignore unreadable files (e.g. binaries)
            }
          }
        }
      };
      await searchDir(path.resolve(dirPath));
      return results.length ? results.join("\n") : "No matches found.";
    } catch (e: any) {
      return `Error finding text: ${e.message}`;
    }
  },
  {
    name: "find_text_in_files",
    description: "Search for specific text across files in a directory.",
    schema: z.object({
      query: z.string().describe("The text to search for"),
      dirPath: z.string().describe("The root directory to search in"),
    }),
  }
);

export const listDirectory = tool(
  async ({ dirPath }) => {
    try {
      const entries = await fs.readdir(path.resolve(dirPath), { withFileTypes: true });
      return entries.map(e => `${e.isDirectory() ? "[DIR]" : "[FILE]"} ${e.name}`).join("\n");
    } catch (e: any) {
      return `Error listing directory: ${e.message}`;
    }
  },
  {
    name: "list_directory",
    description: "List the contents of a directory.",
    schema: z.object({
      dirPath: z.string().describe("The directory path to list"),
    }),
  }
);

export const readProjectStructure = tool(
  async ({ dirPath }) => {
    try {
      const results: string[] = [];
      const buildTree = async (currentPath: string, prefix: string = "") => {
        const entries = await fs.readdir(currentPath, { withFileTypes: true });
        for (let i = 0; i < entries.length; i++) {
          const entry = entries[i];
          const isLast = i === entries.length - 1;
          const marker = isLast ? "└── " : "├── ";
          results.push(`${prefix}${marker}${entry.name}`);
          
          if (entry.isDirectory() && entry.name !== "node_modules" && entry.name !== ".git") {
            const childPrefix = prefix + (isLast ? "    " : "│   ");
            await buildTree(path.join(currentPath, entry.name), childPrefix);
          }
        }
      };
      results.push(path.basename(path.resolve(dirPath)));
      await buildTree(path.resolve(dirPath));
      return results.join("\n");
    } catch (e: any) {
      return `Error reading project structure: ${e.message}`;
    }
  },
  {
    name: "read_project_structure",
    description: "Returns a tree representation of the workspace/directory structure.",
    schema: z.object({
      dirPath: z.string().describe("The root directory to map"),
    }),
  }
);

export const getErrors = tool(
  async ({ dirPath }) => {
    try {
      // Runs typescript compilation to fetch errors.
      const { stdout, stderr } = await execAsync("npx tsc --noEmit", { cwd: path.resolve(dirPath) });
      return stdout || stderr || "No compilation errors found.";
    } catch (e: any) {
      return `Compilation Errors:\n${e.stdout || e.message}`;
    }
  },
  {
    name: "get_errors",
    description: "Get any compile or lint errors in the workspace by running the build tool.",
    schema: z.object({
      dirPath: z.string().describe("The root directory of the project"),
    }),
  }
);

export const applyPatch = tool(
  async ({ filePath, patchContent }) => {
    try {
      const absPath = path.resolve(filePath);
      return new Promise((resolve) => {
        const diffId = Math.random().toString(36).substring(7);
        diffEvents.emit('propose', { id: diffId, file: absPath, oldContent: "", newContent: patchContent, type: 'edit' });
        
        diffEvents.once(`response_${diffId}`, async (accepted: boolean) => {
          if (accepted) {
            await fs.writeFile(absPath, patchContent, "utf8");
            resolve(`Successfully applied patch to ${filePath}`);
          } else {
            resolve(`User REJECTED the patch in ${filePath}`);
          }
        });
      });
    } catch (e: any) {
      return `Error applying patch: ${e.message}`;
    }
  },
  {
    name: "apply_patch",
    description: "Edit text files by executing a diff/patch.",
    schema: z.object({
      filePath: z.string().describe("The absolute path of the file"),
      patchContent: z.string().describe("The full code content or patch representation to write."),
    }),
  }
);
