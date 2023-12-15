<!-- PROJECT LOGO -->
<br />
<p align="center">
  
  <h1 align="center">Paraphrase Your Data</h1> 
  <h3 align="center">Instructions</h3> 

  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About the project 🤷‍♂️
This project utilizes the newest state-of-the-art multililngual LLM to enhance annotated text-based datasets. In short, it does three three things: 1) Paraphrases a list of sentences 2) checks the quality of the paraphrasings and filters out the bad ones and 3) outputs a new dataset consisting of the original sentences + the paraphrased ones.

This is especially useful for low resource languages where large human annotated datasets are sparse and requires a significant amount of resources to create. With this approach we can double, triple or 10x the length of the annotated datasets while staying true to the original labels and thus allow for better finetuning of new models. 

<!-- USAGE -->
## Usage ✅
To use or reproduce the results you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.76.0 (Universal) on a machine running MacOS Ventura 13.4 on a M1 Max chip. The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment for the project.

1. Clone repository
2. Download Mistral LLM
3. Add your data to the ```data``` folder
4. Run setup.sh
5. [Optional] Change arguments

> **Step 1** Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/jorgenhw/NLP_exam_2023.git
cd NLP_exam_2023
```

> **Step 2** Download Mistral locally

This is done by running the following two line in your terminal, one by one

NB: Make sure you're in the root directory (NLP_exam_2023) and that you are inside the virtual environment.

```
huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --local-dir model --local-dir-use-symlinks False
```

This downloads and saves Mistral7B in the 'model' folder.

NOTE: If you're on a machine with GPU (e.g. M1 Macbooks), install the GPU version of Mistral (.GPTQ) instead with the following line:

```
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-DARE-GPTQ --local-dir model --local-dir-use-symlinks False
```

> **Step 3** Add your data to the ```data``` folder

*Correct format of data:*
This script only takes the file format ```.csv```.

Place the data in the ```data``` folder.

> **Step 4** Run ```setup.sh```

To run the program, we have included a bash script that automatically

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
4. Runs the script
5. Deactivates the virtual environment

Run the code below in your bash terminal:

```
bash setup.sh
```

> **Step 5** [Optional]: Change parameters

The following arguments can be changed using ```arparse```. Below is a table descriping the different arguments.

To print the different arguments in the terminal, write:
```
python3 main.py --help
```

| Argument                | Data Type | Default Value                                                         | Description                                                                                         |
|-------------------------|-----------|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `-h, --help`            | N/A       | N/A                                                                   | Show this help message and exit                                                                     |
| `--temperature`         | float     | 0.8                                                                   | Specify the temperature for Mistral                                                                 |
| `--top_p`               | float     | 0.95                                                                  | Specify the top_p for Mistral                                                                       |
| `--top_k`               | int       | 40                                                                    | Specify the top_k for Mistral                                                                       |
| `--max_new_tokens`      | int       | 1000                                                                  | Specify the max number of new tokens for Mistral                                                    |
| `--context_length`      | int       | 6000                                                                  | Specify the context length for Mistral                                                              |

To specify an argument write

```
python3 main.py --temperature 0.6
```
