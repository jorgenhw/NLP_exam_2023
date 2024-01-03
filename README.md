<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://i.imgur.com/20I4D5w.png">
    <img src="https://i.imgur.com/20I4D5w.png" alt="Logo" width=200 height=200>
  </a>
  
  <h1 align="center">BOOSTING LOW RESOURCE LANGUAGES</h1> 
  <h2 align="center"><i>Generating paraphrase for model optimization</i></h2> 
  <h3 align="center">Cognitive Science // Natural Language Processing 2024</h3>


  <p align="center">
    Jørgen Højlund Wibe & Niels Aalund Krogsgaard
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About the project 🤷‍♂️
This project utilizes the newest state-of-the-art multililngual LLM to enhance annotated text-based datasets. In short, it does three three things: 1) Paraphrases a list of sentences 2) checks the quality of the paraphrasings and filters out the bad ones and 3) outputs a new dataset consisting of the original sentences + the paraphrased ones.

This is especially useful for low resource languages where large human annotated datasets are sparse and requires a significant amount of resources to create. With this approach we can double, triple or 10x the length of the annotated datasets while staying true to the original labels and thus allow for better finetuning of new models. 

The repository is part of the exam in Natural Language Processing at Aarhus University. The associated exam paper is in the folder 'exam_paper'.

<!-- ABOUT THE PROJECT -->
#### A note on reproducibility
This repository is set up to seamlessly run paraphrasing with Mistral 7B using the `setup.sh` bash script, tested on both Mac, Linux and Windows (just follow the steps 1-3). 

However, in relation to the exam paper, the steps necesarry to recreate the fine-tuning process are in notebook format and requires one to run the chunks one by one, potentially changing the file paths. Also, if one wishes to do paraphrases using GPT-3 or GPT-4, it is necesarry to add a `.env` file to the main folder with a OpenAI API key inside in the format `OPENAI_API_KEY = "your code"` including the "" signs. After this, one can simply run the `setup_gpt4.sh` bash script from the terminal writing `bash setup_gpt4.sh`.

<!-- USAGE -->
## Usage ✅
To use or reproduce the results you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.76.0 (Universal) on a machine running MacOS Ventura 13.4 on a M1 Max chip. The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```setup.sh``` bash files contains the steps necesarry to create a virtual environment for the project.

1. Clone repository
2. Add your data to the ```data``` folder
3. Run setup.sh
4. [Optional] Change arguments

> **Step 1** Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/jorgenhw/NLP_exam_2023.git
cd NLP_exam_2023
```

> **Step 2** Add your data to the ```data``` folder

*Correct format of data:*
This script only takes the file format ```.csv```.

Place the data in the ```data``` folder.

> **Step 3** Run ```setup.sh```

To run the program, we have included a bash script that automatically

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
4. Downloads Mistral 7B to the `model` folder.
5. Runs the script
6. Deactivates the virtual environment

Run the code below in your bash terminal:

```
bash setup.sh
```

> **Step 4** [Optional]: Change parameters

The following arguments can be changed using ```arparse```. Below is a table descriping the different arguments.

*Make sure you are in the virtual environment before doing so (`venv_NLP_exam`).*

To print the different arguments in the terminal, write:
```
python3 main_mistral.py --help
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

## Folder structure of repo

```
.
├── .gitignore
├── README.md
├── classification_reports_europarl
│   ├── 10DDSC_europarl_HF1x_gpt4paraphrased_plus_org.csv
│   ├── 10DDSC_europarl_only_HF.csv
│   ├── ...
├── classification_reports_twitter
│   ├── 10classification_report_gpt4_plus_org_paraphrasings.csv
│   ├── 10classification_report_only_gpt4_paraphrasings.csv
│   ├── ...
├── data
│   ├── paraphrasings_on_train_2148rows_seed42_1.csv
│   ├── paraphrasings_on_train_2148rows_seed42_2.csv
│   ├── ...
├── main_gpt.py
├── main_mistral.py
├── model
│   └── empty
├── nbs
│   └── finetune.ipynb
├── requirements.txt
├── setup.sh
└── setup_gpt4.sh
├── src
│   ├── classes.py
│   ├── func_gpt.py
│   └── func_mistral.py
```

