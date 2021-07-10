# metaREVEAL

## Create an environment
<code>conda env create -f metaREVEAL.yml</code>

<code>conda activate metaREVEAL</code>

## Run an experiment

<code>python main.py -d autoDL -l any_time_learning -m True</code>

-d: meta-dataset to be run (*autoDL* or *artificial*)

-l: learning style (*any_time_learning* or *fixed_time_learning*)

-m: load trained models (*True* or *False*)
