# ✨ SuperVAD

Open and performant voice activity detector built on transformers. This repository contains code to train neural network.

* ✨ [Demo](https://supervad.korshakov.com)
* 📚 [SuperVAD Dataset](https://github.com/supervad-dataset)
* 🔨 [SuperVAD Typescript Library](https://www.npmjs.com/package/supervad)

## Dataset

A new [dataset](https://github.com/supervad-dataset) was created for this task and you need to download it for training and evaluation. To do so you should execute script:

`bash
./download.sh
`

## Dependencies

You need next python dependencies:
```bash
pip install torch torchaudio tgqm numpy onnx onnxruntime-gpu
```

## Training

For training you need GPU with 16GB+ VRAM, the best card is RTX 4090. This card requires only ~4 hours of training. Training is as simple as executing the `train.py` script. Inside this script you will find hyperparameters that you can tune.

```bash
python3 ./train.py
```

## Export

To export model to ONNX format and final checkpoints for inference, you can execute `export.py` script. Inside the script you would find paths that you can alter to specify required checkpoint
```bash
python3 ./export.py
```

## Evaluation

For evaluation there is `evaluation.ipynb` which compares performance of `webrtcvad`, `silerovad` and trained network as well as last `8` checkpoints.

# License

MIT
