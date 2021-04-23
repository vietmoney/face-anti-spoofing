# Viet Money - Face detection & anti-spoofing

Pure Python - Face detection & anti-spoofing API and CLI.



<img src="images/logo.png" style="zoom:60%;" align="center"/>



## Prerequisite

### Local

- Python 3.6↑: https://www.python.org/downloads/
- Pytorch 1.5.x: https://pytorch.org/get-started/previous-versions/

```sh
# Install Pytorch cuda if using NVIDIA GPU device. Default: CPU device

> pip3 install torch==1.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

	*or*

```shell
# CUDA 10.2
> pip3 install torch==1.5.0

# CUDA 10.1
> pip3 install torch==1.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
> pip3 install torch==1.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

- Python package requirements:

```shell
> pip3 install -r requirements.txt
```

### Docker

- Docker v20.10.5↑: https://docs.docker.com/get-docker/
- Docker Compose v1.28.5↑: https://docs.docker.com/compose/install

---



## Table of Contents

- [Viet Money - Face detection & anti-spoofing](#viet-money---face-detection---anti-spoofing)
  * [Prerequisite](#prerequisite)
    + [Local](#local)
    + [Docker](#docker)
  * [Table of Contents](#table-of-contents)
  * [Features](#features)
    + [Face detection](#face-detection)
    + [Face anti-spoofing detection](#face-anti-spoofing-detection)
  * [Get started](#get-started)
    + [Face detection](#face-detection-1)
    + [**Face anti-spoofing detection**](#--face-anti-spoofing-detection--)
  * [Documents](#documents)
    + [CLI](#cli)
      - [Common options](#common-options)
      - [Face Detection](#face-detection)
      - [Face Anti Spoofing](#face-anti-spoofing)
      - [Host API](#host-api)
    + [Web API](#web-api)
      - [Face detection](#face-detection-2)
- [License](#license)
- [Contact](#contact)


## Features


### Face detection

<img src="images/example_3.jpg" style="max-width:40%;"/>


### Face anti-spoofing detection

<img src="images/example_4.jpg" style="max-width:40%;"/>


## Get started

***NOTE: all method work in RGB pixel format. *(OpenCV pixel format is BGR -> convert before using)***

### Face detection

<img src="images/example_1.jpg" alt=">" style="max-width:40%;" title="face spoofing detected"/>

- Python API
```python
from library.util.image import imread
from library.face_detector import FaceDetector

face_detector = FaceDetector("data/pretrained/retina_face.pth.tar")

image = imread("images/fake_001.jpg") # image in RGB format
faces = face_detector(image)

>>> faces # [[box, score, land_mark]]
[(array([181,   5, 551, 441], dtype=int32), 
  0.99992156, 
  array([[249, 147],
         [412, 145],
         [306, 192],
         [266, 313],
         [404, 311]], dtype=int32))]
```
- CLI


- Web API


### **Face anti-spoofing detection**

- Python API

```python
from library.util.image import imread
from library.face_antspoofing import SpoofingDetector

face_antispoofing = SpoofingDetector("data/pretrained/fasnet_v1se_v2.pth.tar")

>>> face_antispoofing([box for box, _, _ in faces]) # [(is_real, score)]
[(False, 0.5154606513679028)]
```

<img src="images/example_2.jpg" alt="=" style="max-width:40%;"  title="face detected"/>

- CLI


- Web API


## Documents

### CLI

#### Common options

```shell
>  python service.py --help
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
  api       Run service as API
  detect    Detect face in images
  spoofing  Detect spoofing face in images

```

#### Face Detection

```shell
> python service.py detect --help
Usage: service.py detect [OPTIONS] IMAGES...

  Detect face in images

Options:
  -j, --json PATH  Export result to json file
  -q, --quiet      Turn off STD output
  -c, --count      Counting image during process
  -y, --overwrite  Force write json file.
  --help           Show this message and exit.

```

- Input: image's path *(support file globs)*  
  Example: `python service.py detect ./*{.jpg,.png}` - match with any file with extension is `jpg` and `png`.
  
- Output option:
    - `--json PATH`: Export result to JSON file    

    ```json
    {
        "nums": "int",
        "boxes": "List[int]",
        "scores": "List[float]",
        "landmarks": "List[int]"
    }
    ```
  
    - `--quiet` Turn off STD output
    - `--count` Counting image during process
    - `--overwrite` Force write json file.


#### Face Anti Spoofing

```shell
> python service.py spoofing --help
Usage: service.py spoofing [OPTIONS] IMAGES...

  Detect spoofing face in images

Options:
  -j, --json PATH  Export result to json file
  -q, --quiet      Turn off STD output
  -c, --count      Counting image during process
  -y, --overwrite  Force write json file.
  --help           Show this message and exit.

```
- Input: image's path *(support file globs)*  
  Example: `python service.py spoofing ./*{.jpg,.png}` - match with any file with extension is `jpg` and `png`.
  
- Output option:
    - `--json PATH`: Export result to json file    

    ```json
    {
        "nums": "int",
        "is_reals": "List[bool]",
        "scores": "List[float]",
        "boxes": "List[int]"
    }
    ```
  
    - `--quiet` Turn off STD output
    - `--count` Counting image during process
    - `--overwrite` Force write json file.



#### Host API
```shell
> python service.py api --help
Usage: service.py api [OPTIONS]

  Run service as API

Options:
  --host TEXT     API host. Default: localhost
  --port INTEGER  API port. Default: 8000
  --version TEXT  API version.
  --help          Show this message and exit.
```
**_Run with default uvicorn setting_**:

```shell
> python service.py api

INFO:     Started server process [19802]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

**_Support Docker with environment setting_**:

```shell
# edit API config in `.env.example` or container env
> cp .env.example .env
> docker-compose build && docker-compose up -d
```

### Web API

#### Face detection



# License

[Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0](LICENSES)



# Contact

**Author**: Tin Dang

**Email**: tindht@vietmoney.vn

**Website**: [www.vietmoney.dev](www.vietmoney.dev)