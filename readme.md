# Exercises for GEOV212 hydrogeology

This site contains exercises for the course [GEOV212, Hydrogeology](https://www4.uib.no/en/studies/courses/geov212).

Please follow one of the following links to open the exercises in [Google Colab](https://colab.research.google.com/):


* [An introduction to Python and arrays](http://colab.research.google.com/github/ElcoLuijendijk/GEOV212_hydrogeology_exercises/blob/main/introduction_to_python_arrays.ipynb)
* [Exercise 2: Create your own groundwater model](http://colab.research.google.com/github/ElcoLuijendijk/GEOV212_hydrogeology_exercises/blob/main/Exercise_2_steady_state_groundwater_model.ipynb)
* [Exercise 3: Introduction and background](http://colab.research.google.com/github/ElcoLuijendijk/GEOV212_hydrogeology_exercises/blob/main/exercise_3_background.ipynb)
  * Choose one of the following:
  * [Exercise 3a: Pore pressure and seismicity](http://colab.research.google.com/github/ElcoLuijendijk/GEOV212_hydrogeology_exercises/blob/main/exercise_3a_pore_pressure_and_seismicity.ipynb)
  * [Exercise 3b: Resilience of a qanat to drought](http://colab.research.google.com/github/ElcoLuijendijk/GEOV212_hydrogeology_exercises/blob/main/exercise_3b_qanat_resilience.ipynb)
* [Exercise 4: Compaction](http://colab.research.google.com/github/ElcoLuijendijk/GEOV212_hydrogeology_exercises/blob/main/exercise_4_compaction.ipynb)
* Exercise 5: Map-view groundwater model of a catchment of choice in Norway
  * [5a: Collect model data from open data sources](http://colab.research.google.com/github/ElcoLuijendijk/GEOV212_hydrogeology_exercises/blob/main/exercise_5a_model_data.ipynb)
  * [5b: Run and calibrate a steady-state groundwater model](http://colab.research.google.com/github/ElcoLuijendijk/GEOV212_hydrogeology_exercises/blob/main/exercise_5b_gw_model.ipynb)
  * [5c: Groundwater flooding, geohazards and climate change](http://colab.research.google.com/github/ElcoLuijendijk/GEOV212_hydrogeology_exercises/blob/main/exercise_5c_climate_hazard.ipynb)



# Pre-requisites

* An internet connected desktop or laptop
* A google account to run these notebooks in google colab. 
* Alternatively you can run the exercises on your own machine too by first downloading this repository using the `<> Code` and the `Clone` or `Download ZIP options` on the top right of this site, installing a Python and Jupyter notebook environment, such as [Anaconda](https://www.anaconda.com/), an editor like [Visual Studio Code](https://code.visualstudio.com/) and then using this to open and run the exercise notebooks in this repository.


# Running Exercise 5 Locally

Follow the steps below if you plan to run the exercises on your own machine:

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed
- ~3 GB free disk space for the conda environment

## Setup

**1. Create and activate the environment**

```bash
conda env create -f environment.yml
conda activate GEOV212_ex5
```

**2. Register the kernel with Jupyter**

```bash
python -m ipykernel install --user --name GEOV212_ex5 --display-name "Python (GEOV212 Ex5)"
```

**3. Launch JupyterLab**

```bash
jupyter lab
```

## Running the notebooks

Run the notebooks in this order, each one produces files that the next one depends on:

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `exercise_5a_model_data.ipynb` | Download and prepare input data (DEM, geology, hydrology) |
| 2 | `exercise_5b_gw_model.ipynb` | Build and run the steady-state groundwater model |


## MODFLOW 6 executable

The MODFLOW 6 binary (`mf6`) is already included in `tmp_mf6_ex5/bin/`.  
No separate download is needed. If you get a permission error on macOS/Linux, make it executable:

```bash
chmod +x tmp_mf6_ex5/bin/mf6
```

## Notes

- Exercise 5a downloads ~90 MB of bedrock and hydrological data from Norwegian open-data APIs (NGU, NVE). A stable internet connection is required for the first run; subsequent runs reuse cached files.
- The notebooks were developed and tested on the package versions pinned in `environment.yml`. Newer versions may work but are untested.



# Feedback

Please provide feedback on the course, content, delivery or direct questions at any point to elco.luijendijk@uib.no.

If you spot errors or want to make improvements to this course, please submit a pull request.