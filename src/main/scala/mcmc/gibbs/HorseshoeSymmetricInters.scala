package mcmc.gibbs

import java.io.{File, FileWriter, PrintWriter}
import breeze.linalg.{DenseMatrix, DenseVector, max, upperTriangular}
import breeze.numerics.{exp, pow, sqrt}
import scala.math.{log}
/**
 * Variable selection with Horseshoe. Implementation for asymmetric main effects and symmetric interactions.
 * Extends AsymmetricBoth for the main effects and mu, tau, and implements updates for taus and interactions
 * Model: X_ijk | mu,a_j,b_k , tauHS, lambda_jk, theta_jk, tau  ~ N(mu + a_j + b_k + I_jk * theta_jk , τ^−1 )
 * Using gamma priors for taua and taub, Cauchy(0,1)+ for tauHS and lambda_jk
 * Variable selection with Horseshoe: theta_jk| tauHS, lambda_jk ~ N(0 , tauHS^2, lambda_jk^2^ )
 * Asymmetric main effects: as and bs come from a different distribution
 * Symmetric Interactions: I_jk * theta_jk = I_kj * theta_kj
 **/

class HorseshoeSymmetricInters extends HorseshoeAsymmetricBoth {
  private var iterationCount = 0
  override def variableSelection(info: InitialInfo) = {
    // Initialise case class objects
    val initmt = DenseVector[Double](0.0, 1.0)
    val inittaus = DenseVector[Double](1.0, 1.0)
    val initAlphaCoefs = DenseVector.zeros[Double](info.alphaLevels)
    val initBetaCoefs = DenseVector.zeros[Double](info.betaLevels)
    val initZetaCoefs = DenseVector.zeros[Double](info.zetaLevels)
    val initGammas = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels) //Thetas represent the interaction coefficients gamma for this case
    val initLambdas = DenseMatrix.ones[Double](info.zetaLevels, info.zetaLevels)
    val initTauHS = 1.0
    val initAcceptanceCountLambda = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels)
    val initTuningParLambda = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels)
    val initAcceptanceCountTau = 0.0
    val initTuningParTau = 0.0

    val fullStateInit = FullState(initAlphaCoefs, initBetaCoefs, initZetaCoefs, initGammas, initLambdas, initTauHS, initmt, inittaus, initAcceptanceCountLambda, initTuningParLambda, initAcceptanceCountTau, initTuningParTau)
    calculateAllStates(info.noOfIter, info, fullStateInit)
  }

  /**
   * Function for updating indicators, interactions and final interaction coefficients
   */
  override def nextIndicsInters(oldfullState: FullState, info: InitialInfo, inBurnIn: Boolean): FullState = {

    val curGammaEstim = (DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels))
    val curLambdaEstim = (DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels))
    val njk = info.noOfInters //no of interactions
    val batchSize = 50 //batch size for automatic tuning

    var acceptedCountLambda = oldfullState.lambdaCount //counter for the times the proposed value is accepted
    var acceptedCountTauHS = oldfullState.tauHSCount //counter for the times the proposed value is accepted
    iterationCount += 1

    // Update tauHS here because it is common for all gammas
    val oldTauHS = oldfullState.tauHS

    //val stepSizeTauHS = 0.014
    //val stepSizeLambda = 2.5 //sigma

    val oldLambdaTuningParams = oldfullState.lambdaTuningPar
    // Automatic adaptation of the tuning parameters based on paper: http://probability.ca/jeff/ftpdir/adaptex.pdf
    val (stepSizeTauHS, stepSizeLambda, lsiLambda, lsiTauHS) = if (inBurnIn){
      if(iterationCount % batchSize == 0){
        val n = iterationCount / batchSize
        val deltan = scala.math.min(0.01, 1/sqrt(n))

        val accFracHS = acceptedCountTauHS / batchSize
        val accFracLambda = acceptedCountLambda.map(x=> x / batchSize) // ratio of accepted for each lambda as it is DenseMatrix

        acceptedCountTauHS = 0
        acceptedCountLambda = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels) //Reset all counts to 0 since batch is completed

        def checkRatioAndNewDiff(ratio: Double, deltan: Double): Double = {
          if (ratio > 0.44){
            deltan
          }else{
            deltan * (-1)
          }
        }

        val lsiTauHS = if(accFracHS > 0.44){
          oldfullState.tauHSTuningPar + deltan
        }else{
          oldfullState.tauHSTuningPar - deltan
        }
        // Here I have to assign values to lsiLambda
        val lsiLambda = accFracLambda.map(x => checkRatioAndNewDiff(x, deltan)) + oldLambdaTuningParams
        val newLamExp = lsiLambda.map(i=> exp(i))
        (exp(lsiTauHS), newLamExp, lsiLambda, lsiTauHS)
      } else{ // if not batch of 50 completed
        val lsiTauHS = oldfullState.tauHSTuningPar
        val lsiLambda = oldLambdaTuningParams
        val newLamExp = lsiLambda.map(i=> exp(i))
        (exp(lsiTauHS), newLamExp, lsiLambda, lsiTauHS)
      }
    }else{ // if not in burn-in
      val lsiTauHS = oldfullState.tauHSTuningPar
      val lsiLambda = oldLambdaTuningParams
      val newLamExp = lsiLambda.map(i=> exp(i))
      (exp(lsiTauHS), newLamExp, lsiLambda, lsiTauHS)
    }

    //    println(s"stepSizeTauHS", stepSizeTauHS)
    //    println(s"stepSizeLambda", stepSizeLambda)

    // 1. Use the proposal N(prevTauHS, stepSize) to propose a new location tauHS* (if value sampled <0 propose again until >0)
    val tauHSStar = breeze.stats.distributions.LogNormal(log(oldTauHS), stepSizeTauHS).draw()

    // Reject tauHSStar if it is < 0. Based on: https://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
    val curTauHS =if(tauHSStar < 0){
      oldTauHS //Return value to curTauHS
    }
    else {
      val oldTauHSSQR = scala.math.pow(oldTauHS, 2)
      val tauHSStarSQR = scala.math.pow(tauHSStar, 2)

      /**
       * Function for estimating the sqr of two matrices, do elementwise division if the denominator is not 0 and calculate the sum
       */
      def elementwiseDivisionSQRSum(a: DenseMatrix[Double], b: DenseMatrix[Double]): Double = {
        val divMat = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels)
        val aSQR = a.map(x => scala.math.pow(x, 2))
        val bSQR = b.map(x => scala.math.pow(x, 2))
        for (i <- 0 until info.zetaLevels){
          for (j <- 0 until info.zetaLevels){
            if(bSQR(i,j) != 0){
              divMat(i,j) = aSQR(i,j) / bSQR(i,j)
            }
          }
        }
        breeze.linalg.sum(upperTriangular(divMat))
      }

      //2. Find the acceptance ratio A. Using the log is better and less prone to errors due to overflows.
      val A = log(oldTauHSSQR + 1) + njk * log(oldTauHS) - log(tauHSStarSQR + 1) - njk * log(tauHSStar) + 0.5 * elementwiseDivisionSQRSum(oldfullState.gammaCoefs, oldfullState.lambdas) * ((1/oldTauHSSQR) - (1/tauHSStarSQR)) + log(tauHSStar) - log(oldTauHS)

      //3. Compare A with a random number from uniform, then accept/reject and store to curLambdaEstim accordingly
      val u = log(breeze.stats.distributions.Uniform(0, 1).draw())

      if(A > u){
        if(inBurnIn){
          acceptedCountTauHS += 1
        }
        tauHSStar //Return value to curTauHS
      } else{
        oldTauHS //Return value to curTauHS
      }
    }

    info.structureSorted.foreach(item => {
      val j = item.a
      val k = item.b

      // Update lambda_jk
      // 1. Use the proposal N(prevLambda, stepSize) to propose a new location lambda* (if value sampled <0 propose again until >0)
      val oldLambda = oldfullState.lambdas(item.a, item.b)
      val curGamma = oldfullState.gammaCoefs(item.a, item.b)

      val lambdaStar = breeze.stats.distributions.LogNormal(log(oldLambda), stepSizeLambda(item.a, item.b)).draw()

      // Reject lambdaStar if it is < 0. Based on: https://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
      if(lambdaStar < 0){
        curLambdaEstim(item.a, item.b) = oldLambda
      }
      else {
        val oldLambdaSQR = scala.math.pow(oldLambda, 2)
        val lambdaStarSQR = scala.math.pow(lambdaStar, 2)
        val tauHSSQR = scala.math.pow(curTauHS, 2)

        //2. Find the acceptance ratio A. Using the log is better and less prone to errors.
        val A = log(oldLambdaSQR + 1) + log(oldLambda) - log(lambdaStarSQR + 1) - log(lambdaStar) + (scala.math.pow(curGamma, 2)/(2.0 * tauHSSQR)) * ((1/oldLambdaSQR) - (1/lambdaStarSQR)) + log(lambdaStar) - log(oldLambda)

        //3. Compare A with a random number from uniform, then accept/reject and store to curLambdaEstim accordingly
        val u = log(breeze.stats.distributions.Uniform(0, 1).draw())

        if(A > u){
          curLambdaEstim(item.a, item.b) = lambdaStar
          if(inBurnIn){
            acceptedCountLambda(item.a, item.b) += 1
          }
        } else{
          curLambdaEstim(item.a, item.b) = oldLambda
        }
      }

      // Number of the observations that have alpha==j and beta==k and alpha==k and beta==j
      val Njkkj = item.list.length

      // Sum of the observations that have alpha==j and beta==k and alpha==k and beta==j
      val SXjkkj = item.list.sum
      val NoOfajForbk = info.structure.calcAlphaBetaLength(j, k) //No of observations for which a==j and b==k
      val NoOfakForbj = info.structure.calcAlphaBetaLength(k, j) //No of observations for which a==k and b==j

      def returnIfExists(dv: DenseVector[Double], ind: Int) = {
        if (ind < dv.length) dv(ind)
        else 0.0
      }

      val SigmaTheta =
        if (j == k) {
          SXjkkj - Njkkj * oldfullState.mt(0) - NoOfajForbk * (returnIfExists(oldfullState.acoefs, item.a) + returnIfExists(oldfullState.bcoefs, item.b))
        } else {
          SXjkkj - Njkkj * oldfullState.mt(0) - NoOfajForbk * (returnIfExists(oldfullState.acoefs, item.a) + returnIfExists(oldfullState.bcoefs, item.b)) - NoOfakForbj * (returnIfExists(oldfullState.acoefs, item.b) + returnIfExists(oldfullState.bcoefs, item.a))
        }

      val tauGammajk = 1.0 / scala.math.pow(curLambdaEstim(item.a, item.b) * curTauHS, 2)

      val varPInter = 1.0 / (tauGammajk + oldfullState.mt(1) * Njkkj) //the variance for gammajk
      val meanPInter = (info.gammaPriorMean * tauGammajk + oldfullState.mt(1) * SigmaTheta) * varPInter
      curGammaEstim(item.a, item.b) = breeze.stats.distributions.Gaussian(meanPInter, sqrt(varPInter)).draw()
      curGammaEstim(item.b, item.a) = curGammaEstim(item.a, item.b)
      curLambdaEstim(item.b, item.a) = curLambdaEstim(item.a, item.b)
    })
    oldfullState.copy(gammaCoefs = curGammaEstim, lambdas = curLambdaEstim, tauHS = curTauHS, lambdaCount = acceptedCountLambda, lambdaTuningPar = lsiLambda, tauHSCount = acceptedCountTauHS, tauHSTuningPar = lsiTauHS)
  }

  override def getFilesDirectory(): String = "./data/15x20"

  override def getInputFilePath(): String = getFilesDirectory.concat("/simulInterSymmetricInters.csv")

  override def getOutputRuntimeFilePath(): String = getFilesDirectory().concat("/ScalaSymIntersHorseshoe10x15Runtime.txt")

  override def getOutputFilePath(): String = getFilesDirectory.concat("/ScalaSymIntersHorseshoe10x15Results.csv")

  override def printTitlesToFile(info: InitialInfo): Unit = {
    val pw = new PrintWriter(new File(getOutputFilePath()))

    val thetaTitles = (1 to info.zetaLevels)
      .map { j => "-".concat(j.toString) }
      .map { entry =>
        (1 to info.zetaLevels).map { i => "theta".concat(i.toString).concat(entry) }.mkString(",")
      }.mkString(",")

    val lambdaTitles = (1 to info.zetaLevels)
      .map { j => "-".concat(j.toString) }
      .map { entry =>
        (1 to info.zetaLevels).map { i => "lambda".concat(i.toString).concat(entry) }.mkString(",")
      }.mkString(",")

    pw.append("mu ,tau, taua, taub, tauHS,")
      .append(lambdaTitles)
      .append(",")
      .append( (1 to info.alphaLevels).map { i => "alpha".concat(i.toString) }.mkString(",") )
      .append(",")
      .append( (1 to info.betaLevels).map { i => "beta".concat(i.toString) }.mkString(",") )
      .append(",")
      .append(thetaTitles)
      .append("\n")

    pw.close()
  }

  override def printToFile(fullStateList: FullStateList): Unit = {
    val pw = new PrintWriter(new FileWriter(getOutputFilePath(), true))

    fullStateList.fstateL.foreach { fullstate =>
      pw
        .append(fullstate.mt(0).toString)
        .append(",")
        .append(fullstate.mt(1).toString)
        .append(",")
        .append( fullstate.tauab.toArray.map { tau => tau.toString }.mkString(",") )
        .append(",")
        .append( fullstate.tauHS.toString )
        .append(",")
        .append( fullstate.lambdas.toArray.map { theta => theta.toString }.mkString(",") )
        .append(",")
        .append( fullstate.acoefs.toArray.map { alpha => alpha.toString }.mkString(",") )
        .append(",")
        .append( fullstate.bcoefs.toArray.map { beta => beta.toString }.mkString(",") )
        .append(",")
        .append( fullstate.gammaCoefs.toArray.map { theta => theta.toString }.mkString(",") )
        .append("\n")
    }
    pw.close()
  }
}
