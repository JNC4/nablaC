import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: { port: 5102, strictPort: true },
  preview: { port: 5102, strictPort: true },
  worker: { format: 'es' },
  build: { target: 'es2022', outDir: 'dist' },
});
