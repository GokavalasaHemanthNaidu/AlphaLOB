# 🚀 AI SYSTEM DIRECTIVES: ALPHALOB PROJECT

## 1. The Master Blueprint is Law
The absolute source of truth for this entire project is the **AlphaLOB** section in `C:\Users\Hemanth\Downloads\portfolio_blueprint_2027.md`. 
- **DO NOT** hallucinate features, architectures, or tech stacks not mentioned in the blueprint.
- **DO NOT** simplify the system into a basic CRUD app. This is an elite High-Frequency Trading (HFT) stream processor.
- **LOCAL EXCEPTION:** Due to the user's 8GB RAM constraint, we are substituting heavy infrastructure (Kafka, TimescaleDB, Redis) with lightweight local alternatives (`asyncio.Queue`, SQLite, in-memory dicts) for development. All code MUST be written using modular interfaces so the heavy AWS infrastructure can be easily swapped back in for deployment.

## 2. Strict Phased Execution
Never jump ahead. Before writing code for a new phase:
1. Consult the blueprint to understand the exact mathematical and architectural requirements.
2. Generate an `implementation_plan.md` artifact.
3. Wait for the user to explicitly say "approved".

## 3. Quantitative Rigor
This project targets firms like Jane Street, Citadel, and Optiver. 
- All Python code must be strict, typed, and production-grade.
- Mathematical features (WOFI, Kyle's Lambda, Hawkes Process) must be calculated exactly as they would be on a trading desk.
- Focus relentlessly on latency and memory safety (e.g., do not load massive dataframes into memory; process streams chunk by chunk).

## 4. Current Status
- **Phase 1 (Ingestion):** COMPLETE. (See `main.py` and `ingestion_worker.py`).
- **Phase 2:** Microstructure Feature Engineering (Pending).
- **Phase 3:** ONNX Transformer Integration (Pending).
