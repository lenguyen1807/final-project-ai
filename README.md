# Hướng dẫn chạy Medical Image Captioning Pipeline

## Cài đặt dependencies

```bash
pip install -r requirements.txt
pip install tensorboard -U
```

## Các cách chạy

### 1. Chạy nhanh với GPT-2 (đơn giản nhất)

```bash
python quick.py
```

### 2. Chạy với T5 decoder

```bash
python run_t5.py
```

### 3. Chạy với factory pattern

```bash
# GPT-2
python run_factory.py --decoder gpt2

# T5
python run_factory.py --decoder t5

# LLaMA (cần authentication)
python run_factory.py --decoder llama
```

### 4. Chạy với configuration file

```bash
python run_config.py --configs configs/default_config.yaml
```

### 5. Chạy với main script (đầy đủ nhất)

```bash
python main.py --configs configs/default_config.yaml --output my_caption.txt
```

### 6. Chạy với encoder và decoder (mới)

```bash
# ViT + GPT-2
python run_with_encoder.py --encoder vit --decoder gpt2

# Swin + T5
python run_with_encoder.py --encoder swin --decoder t5

# ResNet + GPT-2
python run_with_encoder.py --encoder resnet --decoder gpt2
```

### 7. Test encoder module

```bash
python test_encoder.py
```

### 8. Test full pipeline

```bash
python test_full_pipeline.py
```

## Các tham số có thể sử dụng

### run_factory.py
- `--decoder`: Chọn decoder type (gpt2, t5, llama)
- `--model`: Tên model cụ thể
- `--output`: File output

### run_config.py
- `--configs`: Đường dẫn đến file config YAML
- `--output`: File output
- `--verbose`: Hiển thị thông tin chi tiết

### main.py
- `--configs`: Đường dẫn đến file config YAML (bắt buộc)
- `--output`: File output
- `--verbose`: Hiển thị thông tin chi tiết

## Ví dụ chạy

### Chạy nhanh nhất
```bash
python quick.py
```

### Chạy với config tùy chỉnh
```bash
python run_config.py --configs configs/default_config.yaml --output my_result.txt --verbose
```

### Chạy với T5
```bash
python run_t5.py
```

### Chạy với factory pattern
```bash
python run_factory.py --decoder gpt2 --output gpt2_result.txt
```

## Output

Tất cả các script sẽ:
1. Hiển thị generated caption trên console
2. Lưu caption vào file output
3. Hiển thị thông báo thành công/thất bại

## Troubleshooting

1. **Import errors**: Đảm bảo đã cài đặt tất cả dependencies
2. **CUDA errors**: Sử dụng `device="cpu"` trong config
3. **Model download**: Lần đầu chạy sẽ download models từ HuggingFace
4. **Memory issues**: Sử dụng models nhỏ hơn như `gpt2` thay vì `gpt2-large`

## Cấu trúc file output

```
generated_caption.txt          # Output từ quick.py
t5_generated_report.txt       # Output từ run_t5.py
factory_output.txt            # Output từ run_factory.py
config_output.txt             # Output từ run_config.py
my_caption.txt                # Output từ main.py
```
