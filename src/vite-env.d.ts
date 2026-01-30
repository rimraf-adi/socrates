/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_GROQ_API_KEY: string;
    readonly VITE_SEARXNG_BASE_URL: string;
}

interface ImportMeta {
    readonly env: ImportMetaEnv;
}
