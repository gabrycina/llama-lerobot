# 🤖 Gripmind: LLM based robotics interfaces

An open-source project combining [Llama 3.2 Vision model](https://about.fb.com/news/2024/01/llama-3-now-available/), robotic controls and brain computer interfaces to enable intuitive (and accesible!) human-robot interaction. Built on top of open-source projects including [LeRobot](https://github.com/huggingface/lerobot), [Llama](https://github.com/facebookresearch/llama), and [EMOTIV's Cortex API](https://github.com/Emotiv/cortex-v2-example).

## 🦙 Features 🦙

### 👁️ Vision-Based Spatial Understanding
- Currently powered by Llama 3.2 90B Vision through Groq
- Real-time environment analysis and spatial reasoning
- Action sequence generation based on visual input
- Planned edge deployment using smaller models (1B and 3B parameters)
  - Local inference for improved latency
  - Reduced hardware requirements
  - Offline operation capability

### 🦾 Robotic Control
- Compatible with Moss v1 robotic arm
- Precise motor control through LeRobot integration
- Support for complex manipulation tasks

### 🧠 Brain-Computer Interface
- Direct mind control of robotic arms using EMOTIV EEG headsets
- Real-time neural signal processing
- Built on the open Cortex API for BCI integration

## 🚀 Getting Started

### Prerequisites
- EMOTIV EEG headset
- Moss v1 robotic arm ([assembly instructions](https://github.com/jess-moss/moss-robot-arms))
- Python 3.10+

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/mindgrip.git
cd mindgrip
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables in a global .env with:
```bash
export GROQ_API_KEY="your_key_here"  # Required for Llama 90B model
```

### Hardware Setup

1. Follow the [Moss v1 assembly guide](https://github.com/jess-moss/moss-robot-arms) for robotic arm setup
2. Connect your EMOTIV headset following the [Cortex API documentation](https://emotiv.gitbook.io/cortex-api/)

## 💡 Usage

### Basic Control Flow
```python
from mindgrip.llama import LlamaPolicy
from mindgrip.cortex import CortexInterface

# Initialize components
policy = LlamaPolicy()
bci = CortexInterface()

# Start control loop
while True:
    # Get BCI input
    command = bci.get_command()
    
    # Process with vision system
    action = policy.get_action(command)
    
    # Execute on robot
    robot.execute(action)
```

## 🗺️ Roadmap - It's just the start!

- [x] Initial integration with Llama 3.2 90B Vision
- [ ] Edge deployment with 1B parameter model
- [ ] Edge deployment with 3B parameter model
- [ ] Offline operation support
- [ ] Improved latency through local inference

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is fully open source and licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with these amazing open-source projects:
- [LeRobot](https://github.com/huggingface/lerobot) - Robot control framework
- [Llama](https://github.com/facebookresearch/llama) - Vision and language model (90B parameters)
- [EMOTIV Cortex Examples](https://github.com/Emotiv/cortex-example) - BCI integration
- [Moss Robot Arms](https://github.com/jess-moss/moss-robot-arms) - Hardware design

