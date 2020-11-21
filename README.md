VariableSelectionHorseshoe

This project implements Bayesian variable selection, using the Horseshoe prior approach, on the interactions of a hierarchical two-way Anova model.

There are four implementations:
- Both main and interaction effects are treated as asymmetric
- Both main and interaction effects are treated as symmetric
- Only the interaction effects are considered to be symmetric
- Only the main effects are considered to be symmetric


To run the project set the number of iterations, thinning e.t.c. change the relevant parameters in the file "MainRunner.scala".
To select which of the four implementations you want to run change the object in method "def getVariableSelectionVariant()", e.g. to run the AsymmetricBoth, use the object "myHorseshoeAsymmetricBoth"
Then, go to the corresponding scala file and set the paths, e.g. for the "AsymmetricBoth" case go to the file "HorseshoeAsymmetricBoth.scala" and set the following:
1) getFilesDirectory() //The files directory
2) getInputFilePath() //The relative path to the input file (csv)
3) getOutputRuntimeFilePath() //The relative path to the file that will contain the runtime (txt)
4) getOutputFilePath() //The relative path to the output-sampling results file (csv)

Run the "MainRunner.scala" file
