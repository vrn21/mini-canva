# MarketCanvas-Env
Minimalist 2D design canvas for RL agents and MCP.

A Gymnasium-compliant design environment exposing a design canvas (Text, Shape, Image) to agents via semantic JSON or pixel arrays.

## Core Features
- **Dual Action Spaces:** Semantic (API actions) and Low-level (Computer Use).
- **Hybrid Observations:** Structured DOM-like JSON or raw RGB pixels.
- **Heuristic Reward Engine:** Scores alignment, WCAG contrast, and constraint satisfaction.
- **MCP Native:** Built-in server for LLM tool-calling interaction.

## Quick Start
```bash
# Run the demo
python demo.py

# Start MCP Server
python server.py
```

