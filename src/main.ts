import { config } from 'dotenv';
import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'node:path';
import started from 'electron-squirrel-startup';

// Load environment variables from .env file
config();

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (started) {
  app.quit();
}

// Environment variables
const SEARXNG_BASE_URL = process.env.VITE_SEARXNG_BASE_URL || 'http://localhost:8080';
const GROQ_API_KEY = process.env.VITE_GROQ_API_KEY || '';

// IPC Handlers
ipcMain.handle('search-searx', async (_event, query: string) => {
  try {
    const params = new URLSearchParams({
      q: query,
      format: 'json',
      categories: 'general',
    });

    const response = await fetch(`${SEARXNG_BASE_URL}/search?${params.toString()}`);

    if (!response.ok) {
      throw new Error(`SearXNG search failed: ${response.status}`);
    }

    const data = await response.json();
    return data.results.slice(0, 10).map((result: any) => ({
      title: result.title || 'Untitled',
      url: result.url,
      snippet: result.content || '',
    }));
  } catch (error) {
    console.error('SearXNG error:', error);
    throw error;
  }
});

ipcMain.handle('synthesize-answer', async (_event, query: string, context: string, mode: string) => {
  try {
    const systemPrompt = mode === 'simple'
      ? `You are a helpful research assistant. Based on the provided search results, give a clear, concise, and accurate answer to the user's question. Synthesize information from multiple sources when relevant. Always cite your sources using [1], [2], etc. format. Keep the answer focused and under 300 words. Use markdown formatting.`
      : `You are an expert research analyst. Based on the provided search results, provide a comprehensive, in-depth analysis. Structure your response with clear sections using markdown headers. Synthesize information across multiple sources. Always cite sources using [1], [2], etc. format. Include an executive summary at the start and key takeaways at the end.`;

    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: `Question: ${query}\n\nSearch Results:\n${context}\n\nPlease provide your answer:` },
        ],
        temperature: 0.3,
        max_tokens: mode === 'simple' ? 1024 : 4096,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Groq API failed: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    return data.choices[0]?.message?.content || 'No response generated.';
  } catch (error) {
    console.error('Groq error:', error);
    throw error;
  }
});

// Deep Research Handler - LangChain Agentic research
import { conductLangChainResearch, ResearchProgress, DeepResearchResult } from './services/LangChainAgent';

let mainWindowRef: BrowserWindow | null = null;

ipcMain.handle('deep-research', async (event, query: string): Promise<DeepResearchResult> => {
  const webContents = event.sender;

  const onProgress = (progress: ResearchProgress) => {
    webContents.send('deep-research-progress', progress);
  };

  try {
    const result = await conductLangChainResearch(query, onProgress);
    return result;
  } catch (error) {
    console.error('LangChain research error:', error);
    throw error;
  }
});

const createWindow = () => {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // and load the index.html of the app.
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(
      path.join(__dirname, `../renderer/${MAIN_WINDOW_VITE_NAME}/index.html`),
    );
  }

  // Open the DevTools in development
  if (process.env.NODE_ENV !== 'production') {
    mainWindow.webContents.openDevTools();
  }
};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow);

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
