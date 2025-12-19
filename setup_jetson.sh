#!/bin/bash

# Setup Script for DQN Server on Jetson Xavier

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Setting up DQN Server on Jetson ===${NC}"

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install it first.${NC}"
    exit 1
fi

echo -e "Python 3 is found: $(which python3)"

# 2. Check/Install pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}pip3 not found. Attempting to install...${NC}"
    sudo apt-get update && sudo apt-get install -y python3-pip
fi

# 3. System Dependencies (numpy usually needs these on ARM)
echo -e "${GREEN}Installing system dependencies...${NC}"
sudo apt-get install -y libopenblas-base libopenmpi-dev 

# 4. Install Python Dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
pip3 install -r requirements.txt

# 5. Check PyTorch/CUDA
echo -e "${GREEN}Checking PyTorch installation...${NC}"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Environment setup complete!${NC}"
    echo -e "To start the server run: ${YELLOW}python3 dqn_server.py${NC}"
else
    echo -e "${RED}✗ PyTorch check failed or CUDA not available.${NC}"
    echo -e "${YELLOW}NOTE: On Jetson, you often need to install PyTorch from NVIDIA's repository.${NC}"
    echo -e "If pip install failed, please follow: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
fi
