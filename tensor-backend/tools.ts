import { tool } from "@langchain/core/tools";
import { z } from "zod";
import * as fs from "fs/promises";
import * as path from "path";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

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

export const preciseCodeEdit = tool(
  async ({ filePath, searchBlock, replaceBlock }) => {
    try {
      const absPath = path.resolve(filePath);
      const content = await fs.readFile(absPath, "utf8");
      
      if (!content.includes(searchBlock)) {
        return `Error: The searchBlock was not found exactly in the file. To prevent hallucinated edits, the search block must match the existing code precisely.`;
      }
      
      const newContent = content.replace(searchBlock, replaceBlock);
      await fs.writeFile(absPath, newContent, "utf8");
      
      return `Successfully replaced code block in ${filePath}`;
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
      // Since native patch is complex, this tool is a semantic placeholder for advanced patch tools.
      // In a real scenario, this would use a library like 'diff' to apply the unified patch string.
      // For now, if the patchContent is just exact code, we write it.
      await fs.writeFile(absPath, patchContent, "utf8");
      return `Successfully applied patch to ${filePath}`;
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
