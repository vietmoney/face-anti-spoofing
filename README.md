
# VietMoney - Face Anti Spoofing

This repository make api using for face attendance with anti spoofing.

## Install
```shell
pip3 install -r requirements.txt
```

### Usage

#### Docker as API service:
```shell
docker-compose build && docker-compose up -d
```

#### CLI:
```shell
> python service.py --help
Usage: service.py [OPTIONS] COMMAND [ARGS]...

Options:
  --detector-model TEXT       Face detector model file path
  --detector-threshold FLOAT  Face detector model threshold
  --detector-scale INTEGER    Face detector model scale. >= 240
  --spoofing-model TEXT       Face anti-spoofing file path
  --device TEXT               Device to load model.
  --version                   Show the version and exit.
  --help                      Show this message and exit.

Commands:
  api  Run service as API
```

- CLI

---
Owner: VietMoney  
Author: Tin Dang  
