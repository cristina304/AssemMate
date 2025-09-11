# AssemMate: Graph-Based LLM for Robotic Assembly Assistance

The complete ICRA 2026 code will be released soon.  

Meanwhile, a demo is available. **Due to data confidentiality, only non-existent test data is provided.**  

### Downloads

- **Base model**: [LLAMA-3-8B-4bit](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)  
- **Projector**: [AssemMate Projector](https://huggingface.co/susu0521/AssemMate_projector/tree/main)

### Environment Setup

You can create a conda environment using the provided package list:

```bash
conda create -n AssemMate python=3.9 bzip2 ca-certificates certifi openssl readline sqlite tk xz zlib libffi libgcc-ng libstdcxx-ng -y
conda activate AssemMate
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118   --index-url https://download.pytorch.org/whl/cu118
bash setup_env.sh
```
### Demo

```bash
python infer_and_eval.py \
    --model_name=<PATH_TO_BASE_MODEL> \
    --projector_ckpt_path=<PATH_TO_PROJECTOR_WEIGHTS> \
    --struct_dir=<PATH_TO_KG_DIRECTORY> \  
    # nonexistent_data/test1/kg
    --qa_dir=<PATH_TO_QA_DIRECTORY>        
    # nonexistent_data/test1/qa
```