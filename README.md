# Quantified-Self

Analysis and machine learning models of personal time series data collected over two years.

I recorded my weight and various coarse, easy-to-record variables relating to my diet, sleep, exercise, and quality of
life over the course of two years with the aim of satisfying my personal curiosity about my habits and to see if I could
build an accurate machine learning model built only on coarse grained features, e.g. minutes of exercise, number of
sweets eaten, etc. Since tracking diet via calorie counting and calories burned exercising is involved and annoying, my
hope was to capture enough information with easier-to-record features to be able to predict weight changes with less
hassle. I cataloged this project in four Jupyter notebooks:

[analysis](https://github.com/cbattle12/Quantified-Self/blob/main/analysis.ipynb)

Analysis of several questions I had about my habits (like whether or not sweets seem to be addictive).

[data_preparation](https://github.com/cbattle12/Quantified-Self/blob/main/data_preparation.ipynb)

First of three notebooks chronicling the ML model building process: in this one I explore the data I collected and
prepare it for feeding into my neural network models.

[model_selection](https://github.com/cbattle12/Quantified-Self/blob/main/model_selection.ipynb)

Second of three notebooks chronicling the ML model building process: in this one I perform model selection by doing a
random search of hyper parameter space to find the best parameter set for the models I consider. Note that this notebook
takes a long time to run (on the order of an hour on a so-so laptop) with the default number of search iterations. I've
also erased all the model training outputs to make it more readable.

[model_comparison](https://github.com/cbattle12/Quantified-Self/blob/main/model_comparison.ipynb)

Final notebook chronicling the ML model building process: here I compare the various models against each other and check
to see if I was successful in my goal of building a weight prediction model with my feature set.