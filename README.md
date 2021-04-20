
# VietMoney - Face Anti Spoofing

This repository make api using for face attendance with anti spoofing.

## Install
```shell
pip3 install -r requirements.txt
```
## Usage
### CLI:
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

---
#### Command:

##### Face Detection

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
    - `--json PATH`: Export result to json file    

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
  
---

##### Face Anti Spoofing
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

---

##### API
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
**_Support Docker as API service_**:
```shell
docker-compose build && docker-compose up -d
```

**_Local run_**:
```shell
python service.py api
```

---
Owner: VietMoney  
Author: Tin Dang  
