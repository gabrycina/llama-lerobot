# 🤖 LeRobot: AI-Enhanced Robotic Control

A fork of the Hugging Face LeRobot project, enhanced with advanced AI control capabilities and brain-computer interface integration.

## ✨ New Features

### 🦙 Llama Vision Control
Our robot now leverages Llama's vision capabilities to:
- Analyze the environment through real-time camera feed
- Break down complex tasks into primitive robot functions
- Generate executable action sequences autonomously

### 🎯 Intelligent Policy Selection
- **Voice Interface**: Speak your desired task
- **Scene Understanding**: Real-time environment analysis
- **Smart Policy Selection**: Llama automatically chooses the most suitable pre-trained RL policy
- **Available Policies**:
  - Grab and manipulate objects
  - Pick and place operations
  - Complex manipulation tasks

### 🧠 BCI Integration
Control LeRobot directly with your mind:
- EEG-based interface for direct robot control
- Inclusive design for accessibility
- Real-time neural signal processing

## 🚀 Getting Started

Clone the repository
```bash
git clone https://github.com/yourusername/llama-lerobot.git
cd llama-lerobot
```

Install dependencies
```bash
pip install -r requirements.txt
```

Set up your API keys
```bash
export GROQ_API_KEY="your_key_here"
```


## 📖 Usage

### Voice Control

```python
from mindgrip/llama_policy_selector.py
```

### Start voice interaction

```python
result = listen_to_user()
policy = select_policy(result)
```

### BCI Control
```python
from mindgrip.bci_control import bci_controller
bci_controller().start()
```

