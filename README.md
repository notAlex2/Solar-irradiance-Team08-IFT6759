## Solar Irradiance - IFT6759

This project was created as part of the UdeM course IFT6759 (https://admission.umontreal.ca/cours-et-horaires/cours/IFT-6759/). The objective of this project is to predict present and future (up to 6 hours) solar irradiance at any point on a map of continental united states by only using present and past remote sensing readings from satellites (GOES-13). We propose a machine learning model for prediction of solar irradiance. Refer to the report and presentation included in this reporistory for more details.

### Team 08
* Alexander Peplowski
* Harmanpreet Singh
* Marc-Antoine Provost
* Mohammed Loukili


### To run the evaluation script:

```console
1. cd scripts/
2. Update submit_evalution.sh 
3. sbatch submit_evalution.sh
```
OR
```console
1. cd scripts/
2. Update run_evaluatior.sh 
3. Run run_evaluatior.sh
```

### K-Fold Strategy

* Hold out 1 year of data
* No use of k-fold until pipeline is optimized

### Coding Standards

* Lint your code as per PEP8 before submitting a pull request
* Pull requests are required for merging to master for major changes
* Use your own branch for major work, don't use master
* No large files allowed in git
* Mark task in progress on Kanban before starting work

### To setup a new local environment:

```console
module load python/3.7
virtualenv ../local_env
source ../local_env/bin/activate
pip install -r requirements_local.txt
```

### To setup a new server node environment:

```console
module load python/3.7
virtualenv ../server_env --no-download
source ../server_env/bin/activate
pip install --no-index -r requirements.txt
```
OR, if no requirement.txt file is available:
```console
pip install --no-index tensorflow-gpu==2 pandas numpy tqdm
```

### To evaluate results from server locally using tensorboard:

Run the commands to synchronize data from the server and to launch tensorboard:
```console
./rsync_data.sh
./run_tensorboard.sh
```
Use a web browser to visit: http://localhost:6006/

