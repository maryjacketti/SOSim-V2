# SOSim V2

## Table of Contents
1. [Description](#description)
2. [Installation](#installation)
	1. [Hardware Requirements](#hardware_requirements)
	2. [Software Requirements](#software_requirements)
3. [Running SOSim](#running_SOSim)
4. [Authors](#authors)
5. [License](#license)
6. [Acknowledgement](#acknowledgement)

<a name="descripton"></a>
## Description

This Project iS part of Data Science Nanodegree Program by Udacity in collaboration with BIM Watson Studio platform.
The initial dataset contains users information on the articles reading. 
This project is aim to analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles they will like.

The Project contains two files:

1. 'Recommendations_with_IBM.ipynb'. The Jupyter notebook aims to (1) explore the article data; (2) compute user interactions; (3) deal with 'cold start problem': for new users, make recommendations based on the popular content; (4) use the SVD (singular value decomposition) model to build out a matrix decomposition and predict new articles an individual may interact with.  
2. 'Recommendations_with_IBM.html'. The html version of the 'Recommendations_with_IBM.ipynb' Jupyter notebook. 

<a name="installation"></a>
## Installation

<a name="dependencies"></a>
### Hardware Requirements
SOSim v2 is publicly available via the internet and can be installed on both Macintosh and Microsoft 
Windows 32 or 64 bit. To achieve reasonable performance in terms of computational speed (hours), a 3.0 
GHz processor or better is required. SOSim is written in Python with multiprocessing, so a multicore 
processor is preferred to make the code running faster. The program is 23.8 M. SOSim uses 15 threads 
during multiprocessing. If the user does not have a multicore computer (or has a computer with fewer cores),
you will need to revise the ëSOSimsubmerged.pyí or ëSOSimsunken.pyí code to fit the capabilities of the 
computer. After opening the ëSOSimsubmerged.pyí or ëSOSimsunken.pyí code, the user can search (Ctrl+F) ëmp.Pool(15)í. 
Then, you can change 15 to the number of cores you have or to 2 if you only have a single core computer. For 
example, if your computer is only a one core computer, then set to ëmp.Pool(2)í. 

SOSim can run on a computer with a page file (virtual memory) of minimum 2.3 GB. Nevertheless, 
it is recommended that the memory card is of a minimum of 3.0 GB. Memory requirements of SOSim are 
determined by the fact that Python can allocate memory only up to a total of 2.3 GB, including memory 
required for all machine functions prior to running the model, when implemented on the Windows 32 bit
platform (this limitation is not expected if the model is developed in the future for the Windows 64 bit OS).
The total memory used by all processes before running SOSim is typically about 512 MB on machines not 
having many applications installed and many idle processes to run by default, except for Windows 7 and some 
editions of Windows Vista which may consume up to 1 GB when idle. Therefore, for the majority of spill 
cases to be solved with optimal resolution and including recalculations, it is estimated that a computer
would require an available memory of about 1.7 GB (that is, a difference of about 1.7 GB between the 2.3 GB
limit and the kernel memory taken up by idle processes). Indirect warning messages provided by the GUI will
guide the user in setting the best possible resolution to achieve optimal performance in terms of memory.

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

