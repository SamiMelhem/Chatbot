# Chatbot

## Overview

This Python-based Chatbot project showcases my exploration into the realms of artificial intelligence, specifically the field of natural language processing. It is designed to interact with users in a conversational manner, providing responses based on predefined intents and patterns. A key feature of this project is the custom-built neural network model, consisting of 3 dense layers, which I trained to enable the chatbot to understand and respond effectively to user queries.

## Features
- **Conversational Interface**: Engages users with text-based interactions.
- **Custom Training**: Utilizes a trained model to understand and respond to user queries.
- **GUI Implementation**: Features a graphical user interface for easy interaction.

## Technologies
- **Python**: Primary programming language.
- **Natural Language Processing**: Techniques to process and analyze human language.
- **TensorFlow/Keras**: For building and training the chatbot model.
- **Tkinter**: For the graphical user interface.

## Installation
To set up the Chatbot project on your local system, follow these steps:
```bash
git clone https://github.com/SamiMelhem/Chatbot.git
cd Chatbot
pip install tensorflow keras
```

## Usage
Train the chatbot to prep the chatbot for interaction:
```bash
python train_chatbot.py
```

Run the GUI application to interact with the chatbot:
```bash
python gui_chatbot.py
```

## File Descriptions
- `gui_chatbot.py`: The main script to launch the chatbot interface.
- `train_chatbot.py`: Script to train the chatbot model.
- `chatbot_model.h5`: Saved chatbot model.
- `classes.pkl, words.pkl`: Pickle files for classes and words.
- `intents.json`: Contains predefined patterns and responses.
