
### 两个conda环境的创建
```python
python=3.10.12
conda create -n vllm python=3.10
conda create -n alfworld python=3.10
```
### 下载依赖与模型
**windows环境的alfworld缺少依赖，服务器下载速度满，此处建议在wsl配置代理后下载并将依赖下载后scp到服务器，配置好ALFWORLD_DATA路径即可**
```python
pip install -r requirements.txt
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct
```

### 进入vllm环境并运行推理服务
```python
conda activate vllm
```
```bash
cd scripts
bash serve.sh
```
### 进入alfworld并运行评测脚本
```bash
python react/*.py --seed=随机种子
```
traj会保存到<br>
```bash
result/*/*.jsonl
```
