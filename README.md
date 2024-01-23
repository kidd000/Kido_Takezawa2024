# Coevolution of norm psychology and cooperation through exapted conformity

Source Code of Y. Kido and M. Takezawa "Coevolution of norm psychology and cooperation through exapted conformity" in preparation.

## Requirements

To run the simulation, first install the packages listed in requirements.txt.

```install required packages
pip install -r requirements.txt
```

## Models

|                                    |       Model 1        |         Model 2         |         Model 3         |
| ---------------------------------- | :------------------: | :---------------------: | :---------------------: |
| Assumed norm psychology            |      $\alpha_i$      | $\alpha_i$ & $\alpha_d$ | $\alpha_i$ & $\alpha_d$ |
| Initial distribution of $\alpha_i$ | $\sim U(0.00, 0.05)$ |  $\sim U(0.00, 0.05)$   |  $\sim U(0.00, 0.05)$   |
| Initial distribution of $\alpha_d$ |                      |  $\sim U(0.00, 0.05)$   | $\sim N(0.30, 0.25^2)$  |

## Model 1

First, we expand the model of Gavrilets & Richerson (2017), manipulate the content of the injunctive norm exogenously, and explore the conditions of the injunctive norm that promote the co-evolution of norm psychology and cooperation.

To get the results, run [Model1.py](Main/Model1-oneNormPsych.py). The command line argument is the run number of the simulation.

When you run the simulation, directories containing csv files with the frequency of each strategy for the final round, average fitness, and average injunctive norm psychology value of all generations are generated for the combination of normative values of the injunctive norm (11×11=121 combinations).

## Model 2

Next, we expand Model 1 and assume a descriptive norm and its corresponding norm psychology.

To get the results, run [Model2.py](Main/Model2-twoNormPsych.py). The command line argument is the run number of the simulation.

When you run the simulation, the results for Model 1 with csv files written out with the average descriptive norm psychology values are generated.

## Model 3

Finally, we assume the exaptation of the descriptive norm psychology that produces conformity. Specifically, we set the initial distribution of descriptive norm psychology to $\sim N(0.3, 0.25^2)$.

Note that we compare the results of the initial distribution mean $\{0.1, 0.3, 0.5\}$ in Supplementary Information.

To get the results, run [Model3.py](Main/Model3-twoNormPsych-exapt.py). For the command line arguments, give the mean value of the initial distribution in the first, and the run number of the simulation in the second.

When you run the simulation, the same results as Model 2 are generated.

## Supplementary Models

In the Supplementary Information, we mainly conducted a sensitivity check on the following three assumptions.

To run the simulation of each supplementary model, follow the instructions in each section.

### 1. Assumption of intergenerational transmission of strategies

In the Main, we reported the results of assuming that strategies (phenotypes) are not inherited between generations, unlike genotypes (norm psychologies).

In the Supplementary, we reported the results of assuming that strategies are vertically transmitted between generations.

To get the results, run [vertical_transmission_Model1.py](Supplementary/VT-Model1-oneNormPsych.py)， [vertical_transmission_Model2.py](Supplementary/VT-Model2-twoNormPsych.py)， [vertical_transmission_Model3.py](Supplementary/VT-Model3-twoNormPsych-exapt.py).

### 2. Assumption of group size

In the Main, we reported the results of group size 16.

In the Supplementary, we reported the results of assuming group size 8 and 24 in addition.

To get the results of changing the group size, change the value of the `GROUP_SIZE` parameter in the global variables in the Main code and run it.

### 3. Assumption of migration rate

In the Main, we reported the results of migration rate 0.5.

In the Supplementary, we reported the results of assuming migration rate 0.25 and 0.75 in addition.

To get the results of changing the migration rate, change the value of the `MIGRATION_RATE` parameter in the global variables in the Main code and run it.
