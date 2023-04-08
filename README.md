## Tabular Data GAN for Rossmann store sales data

This repository implements a simple GAN model to generate synthetic tabular data which looks like sales data in Rossman Stor Sales dataset.

### Rossmann Store Sales dataset
[Rossmann Store Sales dataset](https://www.kaggle.com/competitions/rossmann-store-sales/overview) contains ~1M daily sales data over 1,115 Rossmann drug stores. It also contains relevant metadata such as state holidays, school holidays, promotions, competitions in the neighborhood. In this repository, I use two following files:
* store.csv: Store information of 1,115 stores.
* train.csv: Daily sales data of stores in store.csv from Jan 2013 to July 2015.

For further information of the dataset, see [here](https://www.kaggle.com/competitions/rossmann-store-sales/data).

As processing the time related data, fields with periodic nature are converted into the cosine and sine values (ex: day of week, day of year). Unranked categorical variables are converted to one-hot vectors, including the store id. Unbounded numerical values are normalized and their stats are stored to restore the tabular data later.

### Models
Both of generator and discriminator uses deep neural net model where each layer is composed of fully-connected layer and batch normalization. Due to relatively small size of features, simple fully-connected layers can do the job.

### Evaluation
Rossmann store sales dataset was proposed on kaggle originally for the regression task on the Sales field. Therefore, to make a generator which can help to synthesize extra training data for this regression task, it is natural to evaluate the generator model by the relative performace on this regression task compared to the training with real data. For the regression metric, I use Root Mean Square Percentage Error (RMSPE), which is calcualted as:

$$RMSPE=\sqrt{\frac{1}{\text{# of non-zero }x_i} \sum_{i, \ x_i\ne 0} \left(\frac{\hat{x}_i - x_i}{x_i}\right)^2 }$$

For simple and quick evaluation, I use the simple linear regression model. As the final metric, ratio of the synthetic-data-trained model's RMSPE to the real-data-trained model's RMSPE will be reported. The closer the ratio to 1, the better generator (much larger than 1 if the generator is poor).

For further customization, I also made options to choose decision tree regressor insted of linear regression model, RMSE instead of RMSPE, and the margin between two metrics instaed of the ratio between two metrics.


### How to use
To build a docker image, run the following ('tdg' is the image name):
```
docker build . -t tdg
```
Once the image built, run the following to start the container and access the command line in that container:
```
docker run -it --rm -v .:/workspace tdg
```
Once the container is created and got access to its command line, one can run following commands within the container.

#### Combine store.csv & train.csv and build train/dev/test sets
```
python3 -c "import dataset; dataset.split_sales_dataset('data/store.csv', 'data/train_tiny.csv', 'DIRECTORY_TO_STORE_FILES', 'DATASET_NAME', [0.9, 0.05, 0.05])"
```
Customize DIRECTORY_TO_STORE_FILES and DATASET_NAME. Also one can customize train/dev/test set splits by modifying [0.8, 0.1, 0.1].

#### Train model
```
python3 run_experiment.py [TRAINING_CONFIG_FILE]
```
A sample TRAINING_CONFIG_FILE can be found in `configs/sample.yaml`. Following is the description of some fields in the training config file:
* store_fmt: Store id format. 'onehot' or 'numeral'.
* dayofweek_fmt: Day of week format. 'periodic' or 'onehot'.
* date_fmt: Date format. 'periodic' or 'numeral'.
* noise_size: Size of noise to add to the real data vector.
* latent_dim: Dimension of random field to feed to the generator.
* g_hdims, d_hdims: Hidden layer sizes for generator/discriminator. Given in the list format.
* g_lr, d_lr : Learning rate for generator/discriminator.
* g_beta1, g_beta2, d_beta1, d_beta2: Betas for Adam optimizer.
* model_dir: Directory to save the generator model. Not saving if not specified.
* log_dir: Directory to save training log. Not saving if not spacified.

#### Evaluate model
```
python3 run_evaluation.py -d [DATA_FILE_PATH] -m [MODEL_PATH] -c [CONFIG_PATH]
```
Here, CONFIG_PATH is the path of the config file used to train the specified model. A sample model file for `configs/sample.yaml` can be found in `models/sample.pt`. For furhter optional arguments, run `python3 run_evaluation.py --help`.

#### Generate synthetic table
```
python3 generate_table.py [MODEL_PATH] [CONFIG_PATH] [TABLE_SIZE] [CVS_PATH]
```
TABLE_SIZE is the number of rows to synthesize. CSV_PATH is the path to save the table as a csv file.