# Feature-Swapped Multimodal Reasoning Framework for PMR

## Fold Structure
 ``` bash
FSMR
├── a_transformers #code
├── clip #code
├── data_processing #code
├── local_transformers #code and the roberta-large checkpoint
├── modeling #code
├── utils #code
├── output #output path (generated after running)
│   └── checkpoint #checkpoints and log files
│   └── results #test result jsons
├── Checkpoints_and_Data #checkpoints and pre-processd PMR dataset
├── Multi-View-Reasoning-cold-start-1.pth #cold-start checkpoint
├── config.yaml #configuration
└── run_PMR_FSMR.py
 ``` 


## Training
1. modify config.yaml

    - set _**do_train**_ to _**True**_

    - modify _**output_dir**_

    ```bash
    # modify config.yaml as this
    do_train: True
    output_dir: ./output/checkpoint/output1/
    ```



2. run
    ```bash
    python run_PMR_FSMR.py
    ```


## Testing
1. modify config.yaml


    - set _**do_train**_ to _**False**_
    
    - modify _**eval_model_dir**_

    - modify _**result_dir**_

    ```bash
    # modify config.yaml as this
    do_train: False
    eval_model_dir: ./output/checkpoint/output1/FSMR-2-0.6326397919375812-1600.pth
    result_dir: ./output/results/output1/
    ```


2. run
    ```bash
    python run_PMR_FSMR.py
    ```


