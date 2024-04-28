# AI Chatbot

## Overview
This repository contains chatbot programs designed to use large language models.

## Repository Structure
Each specific use case is under its own folder and are named accordingly.

## Environment Setup
First, clone the repository.
```
git clone https://github.com/kaelfdl/ai-chatbot
cd ai-chatbot
```

Next, create a python virtual environment and install the dependencies.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Example
For instance, we'll run the chatbot program.
```
cd chatbot_pytorch
python chat.py
Let's chat! Type 'quit' to exit, or type a goodbye message
You: Hello
🐱Cy: Hi there, what can I do for you?
You: What items do you sell?
🐱Cy: We have coffee and tea
You: Can I pay by cash?
🐱Cy: We accept VISA, Mastercard and Paypal
You: ok then, bye
🐱Cy: See you later, thanks for visiting
```
