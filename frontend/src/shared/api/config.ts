export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1';

const parsedTopK = Number(import.meta.env.VITE_RAG_TOP_K ?? 20);
export const RAG_DEFAULT_TOP_K = Number.isFinite(parsedTopK) && parsedTopK > 0 ? Math.floor(parsedTopK) : 20;
export const RAG_DEFAULT_SOURCE_ID =
  import.meta.env.VITE_RAG_SOURCE_ID ?? 'ecfr-title-12-chapter-xii';
