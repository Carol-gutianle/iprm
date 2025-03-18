# MLLM_PRIME
This project implements the implicit process reward model (PRIME) training for multimodal large language models (MLLMs).

## Features
MLLM_PRIME is built upon **openrlhf==0.6.1.post1**, which you can download from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). The uni-modal implementation can be found in [ImplicitPRM](https://github.com/PRIME-RL/ImplicitPRM).  

I opted not to use **openrlhf==0.4.4** in this implementation of ImplicitPRM, as it lacks support for many recent Vision-Language Models. However, you can replace the OpenRLHF version in this repository as needed to ensure compatibility with your policy model.

## Usage

### Train Implicit Process Model
```bash
bash run_ce.sh
```

### Evaluate
To evaluate the model, refer to `openrlhf/cli/eval_ce.py`. Ensure you modify the `convert_data` function to align with the data format used in your `CEMLLMDataset`.
```bash
bash eval_prm.sh
```

## BugList
* Do not use Python==3.9.0 and 3.9.1