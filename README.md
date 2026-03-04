# Hybrid ML & DL Methods for Cyberbullying Detection

This project applies Artificial Intelligence methods, specifically Hybrid Machine Learning (ML) and Deep Learning (DL), for detecting cyberbullying in text data (tweets/texts).

## Requirements

We use Python 3.11.5.

All Python packages needed are listed in the `requirements.txt` file and can be installed using the pip command. 

Assuming you have `python` and `pip` installed, third-party packages can be installed with:

```bash
pip install -r requirements.txt
```

## How to run on a particular dataset

The program is prepared to use specific predefined datasets located in the `Datasets/` directory and generated data in `Generate Datasets/`.

To classify data and run experiments, run the `run.py` script specifying the model type and dataset name:

```bash
python run.py <model_type> <data_name>
```

### Parameters

**`<model_type>`**
- `ml`: Run Machine Learning experiments (e.g., SVM, LDA, SVM-LDA).
- `dl`: Run Deep Learning experiments (e.g., CNN, GRU, combinations).

**`<data_name>`**
- `dataset1`: Uses `dataset1 - twitter_racism_parsed_dataset.csv`.
- `dataset2`: Uses `dataset2 - cyberbullying_tweets.csv`.
- `dataset3`: Uses `dataset3.csv`.
- `simdataset`: Uses JSON files from the `Generate Datasets` directory.

### Example Usage

To run Deep Learning models on dataset 1, use:

```bash
python run.py dl dataset1
```

To run Machine Learning models on the simulated dataset, use:

```bash
python run.py ml simdataset
```

## Results

After the completion of the experiments, the results will be saved automatically as an Excel file in the main directory in the format: `Results_<model_type>_<dataset>.xlsx`.


## 📄 Citation

If you use this code or the methodology implemented here for your research (specially Hybrid Machine Learning), please cite the following paper:

### APA Format
Syafariani, F., Lola, M. S., Mutalib, S. S. S. A., Nasir, W. N. F. W., Hamid, A. a. K. A., & Zainuddin, N. H. (2025). Leveraging a hybrid machine learning model for enhanced cyberbullying detection. Aptisi Transactions on Technopreneurship (ATT), 7(2). https://doi.org/10.34306/att.v7i2.536


### BibTeX
```bibtex
@article{Syafariani_Lola_Mutalib_Nasir_Hamid_Zainuddin_2025, title={Leveraging A Hybrid Machine Learning Model for Enhanced Cyberbullying Detection}, volume={7}, url={https://att.aptisi.or.id/index.php/att/article/view/536}, DOI={10.34306/att.v7i2.536}, number={2}, journal={Aptisi Transactions on Technopreneurship (ATT)}, author={Syafariani, Fenny and Lola, Muhamad Safiih and Mutalib, Sharifah Sakinah Syed Abd and Nasir, Wan Nuraini Fahana Wan and Hamid, Abdul Aziz K. Abdul and Zainuddin, Nurul Hila}, year={2025}, month={Apr.}, pages={371-386} }
