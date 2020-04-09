package mcmc.gibbs

import breeze.linalg.{*, DenseMatrix, DenseVector}
import structure.DVStructure

case class FullState(acoefs: DenseVector[Double], bcoefs: DenseVector[Double], zcoefs: DenseVector[Double], thcoefs: DenseMatrix[Double], indics: DenseMatrix[Double], finalCoefs: DenseMatrix[Double], mt: DenseVector[Double], tauabth: DenseVector[Double])

case class FullStateList(fstateL: Vector[FullState])

case class InitialInfo(noOfIter: Int, thin: Int, burnIn: Int, N: Int, SumObs: Double, structure: DVStructure, structureSorted: DVStructure, alphaLevels: Int, betaLevels: Int, zetaLevels: Int, noOfInters: Int, sizeOfDouble: Int, alphaLevelsDist: Int, betaLevelsDist: Int,
                       alphaPriorMean: Double, betaPriorMean: Double, thetaPriorMean: Double, mu0: Double, tau0: Double,
                       a: Double, b: Double, aPrior: Double, bPrior: Double, p: Double)