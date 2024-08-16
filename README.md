# U-Net:Oil Spill

- [Quick start](#quick-start)
  - [Windowms 10](#without-docker)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Weights & Biases](#weights--biases)
- [Pretrained model](#pretrained-model)
- [Data](#data)

## Quick start

### Windowms 10

python 3.9版本

1. [下載 CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Install tensorflow-gpu
```
pip install tensorflow-gpu==2.10.1
```
5. Download the data and run training:
```bash
python train.py --amp
```

## Usage

### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

預設情況下，`scale`為 0.5，因此如果您希望獲得更好的結果（但使用更多記憶體），請將其設為 1。

自動混合精度也可透過 `--amp` 標誌使用。[Mixed precision](https://arxiv.org/abs/1710.03740)允許模型使用更少的內存，並且透過使用 FP16 演算法在最新的 GPU 上速度更快。建議啟用 AMP。

### Prediction

訓練模型並將其儲存到 `MODEL.pth` 後，您可以透過 CLI 輕鬆測試影像上的輸出遮罩。

要預測單一圖像並儲存它：

`python predict.py -i image.jpg -o output.jpg`

要預測多個圖像並顯示它們而不保存它們：

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```

您可以透過 `--model MODEL.pth` 指定要使用的模型檔案。

## Weights & Biases

訓練進度可以使用[Weights & Biases](https://wandb.ai/)即時視覺化。損失曲線、驗證曲線、權重和梯度直方圖以及預測光罩都記錄到平台上。

啟動訓練時，控制台中將列印連結。單擊它即可轉到您的儀表板。如果您已有 W&B 帳戶，則可以透過設定 `WANDB_API_KEY` 環境變數來連結它。如果沒有，它將建立一個匿名運行，並在 7 天後自動刪除。

## Data

# 主要檔案為 data/imgs 跟 data/masks(其他為測試程式用資料集)

缺乏一個全面的SAR圖像資料集，所有圖元都適當標記，這是溢油檢測的主要挑戰之一，導致在比較文章中描述的方法時結果不一致。2019年，作者K restenitis從不同的圖像區域創建了一組大致全面的標記資料。該資料庫通過使用歐洲航天局(ESA)的哥白尼開放獲取中心 1 收集石油污染海洋區域的衛星圖像。CleanS eaN et 服務還提供了來自歐洲海事安全局(EMSA )的污染事件的地理座標和時間資訊。其中包括 2015 年 9 月 28 日至 2017 年 10 月 31 日期間的石油洩漏事件，圖像由歐洲衛星 Sentinel-1 在 c 波段以 VV 偏振獲取。在確定感興趣的區域後，對原始 SAR 圖像進行預處理，即將圖像調整為 1250×650 維數，使用 7×7 中心濾波器降低光譜雜訊，然後進行 dB 到亮度的線性轉換。創建了包含 1112 張圖像的，這些圖像被分為5類:石油洩漏、相似物、船舶、陸地和海洋(作為背景)，這些圖像可以通過實驗室的網站公開獲得。
在本研究中，我們使用上述資料集對SAR圖像進行語義分割。常見的比率是80/20,70/30和90/10，但它是任意的。值得注意的是，驗證集主要表示所有輸入範圍內的目標函數。有時驗證集中有簡單的資料，但訓練集中有非常困難的資料，導致關於良好泛化的錯誤陳述。測試集和驗證集約占訓練集的 1 0 -1 5 %，因此將資料集分為三部分，分別為 890 張、110 張和 112 張圖像，分別用於訓練、測試和驗證。


[資料來源](https://github.com/milesial/Pytorch-UNet)
