
Machine Learning Engineer (RL/Agents) Take-Home Assignment
Background
Our team is building foundational LLM agents that autonomously operate sales, marketing, and design software (like Canva, Figma, and Google Workspace) via direct computer use. To train these models using Reinforcement Learning (e.g., PPO, DPO, or ReST), we need robust, scalable simulation environments.
The Objective
Your objective is to build MarketCanvas-Env, a minimalist, deterministically simulated 2D design canvas (a "Mini-Canva"). This environment must expose a standard RL interface (State, Action, Reward) and simultaneously act as a Model Context Protocol (MCP) server so that modern LLMs can natively interact with it via tool calling.
You are not expected to train a model for this assignment. Your focus is strictly on Environment Design, MDP Formulation, and API Tooling.
Core Requirements
1. The Simulator (Mini-Canva)
Build a lightweight 2D canvas engine in Python (e.g., 800x600 resolution).
The canvas should support at least three element types: Text, Shape (Rectangle/Button), and Image (can be represented by colored bounding boxes for simplicity).
Elements should have properties: x, y, width, height, z-index, color/text_color, and content.
2. Observation Space (State)
How does the LLM "see" the canvas? You must implement the state representation:
Semantic State: A JSON representation (DOM tree / Accessibility tree) of the canvas detailing all elements, their properties, and spatial relationships.
Visual State (Optional / Bonus): A rendered pixel array (RGB) of the current canvas, allowing for multimodal agent training in the future.
3. Action Space
Define how the agent interacts with the canvas. You should implement one of the following action spaces (or both, if you want to discuss the trade-offs in your write-up):
Low-Level (Computer Use): mouse_move(x, y), mouse_click(), mouse_drag(x1, y1, x2, y2), keyboard_type(string).
High-Level (Semantic UI): add_element(type, content), move_element(id, new_x, new_y), change_element_color(id, hex_code).
4. Reward Function (The core challenge)
In RL, the reward function dictates the agent's behavior. We want the agent to design a valid marketing asset based on a target prompt (e.g., "Create a Summer Sale email banner with a headline, a yellow CTA button, and good contrast").
Design a heuristic-based reward function that calculates a scalar reward (-1.0 to 1.0) at the end of an episode. Consider scoring based on:
Constraint Satisfaction: Are the required elements (Headline, CTA) present?
Aesthetics/Design Rules: Are elements overlapping illegibly? Are they aligned (e.g., centered)?
Accessibility: Does the text contrast against its background pass basic WCAG ratios?
5. Model Context Protocol (MCP) Integration
Wrap your environment in an MCP Server.
Expose tools like get_canvas_state, execute_action, and get_current_reward.
This allows an LLM client (like Claude Desktop) to connect to your environment and attempt to design a banner using standard tool-calling capabilities.
Deliverables
Please submit a link to a Git repository containing the following:
Python Source Code: Clean, well-documented code for the environment (feel free to use gymnasium structures) and the MCP server.
Demo Script (demo.py): A script that initializes the environment, accepts a mock target prompt, takes a few programmatic (or random) steps, and prints the resulting state and reward.
WRITEUP.md: A 1-2 page document covering:
Your reasoning behind the chosen Action and State spaces.
How your reward function works and its potential loopholes (reward hacking).
Scaling Question: If we wanted to run PPO on this environment with 10,000 parallel rollouts using a Vision-Language Model, what infrastructure bottlenecks would you anticipate, and how would you redesign the environment to handle them?
(Optional) Visual Output: A simple mechanism to save the final canvas state as a .png or draw it using a simple UI library (e.g., Pygame, Tkinter, or just PIL).


