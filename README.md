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
2. Add your data to the ```data``` folder
3. Run setup.sh
4. [Optional] Change arguments

> **Step 1** Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/jorgenhw/NLP_exam_2023.git
cd NLP_exam_2023
```

> **Step 1.2** Download Mistral locally

This is done by running the following line in your terminal

```
huggingface-cli download TheBloke/OpenHermes-2.5-Mistral-7B-GGUF openhermes-2.5-mistral-7b.Q4_K_M.gguf --path ./model --path-use-symlinks False
```
This locates the model in the `model` folder.

> **Step 2** Add your data to the ```data``` folder

This script only takes the file format ```.csv```.

*Correct format of data:*

1. The data can be both long and short format as long as **the first column is the strings to be paraphrased**.
2. Place the data in the ```data``` folder.

> **Step 3** Run ```setup.sh```

In your terminal write

```
bash setup.sh
```

> **Step 4** [Optional]: Change arguments

The following arguments can be changed using ```arparse```. Below is a table descriping the different arguments.

```
bash setup.sh --argument_x_and_y example
```

|    | Argument | Default | Description |
|:------:|:----------:|:------------------:|:------------------:|
|  1  |    39 M    |     `tiny.en`      |       `tiny`       |
|  2  |    74 M    |     `base.en`      |       `base`       |
| 3  |   244 M    |     `small.en`     |      `small`       |
| 4 |   769 M    |    `medium.en`     |      `medium`      |
| 5  |   1550 M   |        N/A         |      `large`       |


