# Local SLM TypeScript Studio — Gemini Agent Context

See `/AGENTS.md` for the complete architecture specifications, directory map, engine mechanics, and persistent operational rules for this project.

### Quick Architecture Summary
- **App**: Browser-native Small Language Model (SLM) in TypeScript/React 19/Vite/Tailwind CSS.
- **Engine**: Pure `Float32Array` scratch-buffered causal transformer (`src/slm/transformer.ts`) + statistical smoothed trigram memory layer (`src/slm/ngram.ts`).
- **Zero Heavy ML Dependencies**: Runs directly on client JavaScript runtime without WebGPU or external frameworks.
- **State**: Centralized in `src/App.tsx`; vocabulary expansion synced across tokenizer and model instances.
