# SOSim-New-Code

![Intro Pic]

## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Libraries](#library)
	2. [Installing](#installing)
	3. [Instruction](#executing)
	4. [Data distribution](#material)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)

<a name="descripton"></a>
## Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with BIM Watson Studio platform.
The initial dataset contains users information on the articles reading. 
This project is aim to analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles they will like.

The Project contains two files:

1. 'Recommendations_with_IBM.ipynb'. The Jupyter notebook aims to (1) explore the article data; (2) compute user interactions; (3) deal with 'cold start problem': for new users, make recommendations based on the popular content; (4) use the SVD (singular value decomposition) model to build out a matrix decomposition and predict new articles an individual may interact with.  
2. 'Recommendations_with_IBM.html'. The html version of the 'Recommendations_with_IBM.ipynb' Jupyter notebook. 

<a name="getting_started"></a>
## Getting Started

<a name="dependencies"></a>
### Libraries
* Machine Learning Libraries: NumPy, Pandas
* Database Libraqries: pickle, project_tests
* Web App and Data Visualization: matplotlib.pyplot
<a name="installing"></a>
### Installing
Clone this GIT repository:
```
https://github.com/jichaojoyce/Recommendations-with-IBM.git
```
<a name="Instruction"></a>
### Instructions:
1. In a terminal or command window, navigate to the top-level project directory Recommendations-with-IBM/ (that contains this README) and run one of the following commands:
```ipython notebook Recommendations_with_IBM.ipynb```

or

```jupyter notebook Recommendations_with_IBM.ipynb```

This will open the iPython Notebook software and project file in your browser.
### Data distribution:
The below figure is the distribution of articles

![dis Pic](DataDistribution.png)

Overall, the total count number is 5148 and 50% of individuals interact with 3 number of articles or fewer. 

<a name="authors"></a>
## Authors

* [Chao Ji](https://github.com/jichaojoyce)

<a name="license"></a>
## License
Feel free to use it!
<a name="acknowledgement"></a>
## Acknowledgements

Credicts give to [Udacity](https://www.udacity.com/).

