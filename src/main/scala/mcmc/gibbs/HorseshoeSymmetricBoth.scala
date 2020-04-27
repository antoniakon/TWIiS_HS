package mcmc.gibbs

import java.io.{File, FileWriter, PrintWriter}
import breeze.linalg.{DenseMatrix, DenseVector, max, upperTriangular}
import breeze.numerics.{exp, pow, sqrt}
import scala.math.{log}

/**
 * Variable selection with Horseshoe. Implementation for symmetric main effects and symmetric interactions.
 * Extends SymmetricMain for updating z coefficients, mu and tau. Implements update for taus and interactions based on SymmetricInteractions class.
 * Model: X_ijk | mu,a_j,b_k ,tauHS, lambda_jk, gamma_jk,tau  ~ N(mu + z_j + z_k + gamma_jk , τ^−1 )
 * Using gamma priors for taua and taub, Cauchy(0,1)+ for tauHS and lambda_jk
 * Symmetric main effects: zs come from the same distribution
 * Symmetric Interactions: gamma_jk = gamma_kj
 **/

class HorseshoeSymmetricBoth extends HorseshoeSymmetricMain {
  override var iterationCount = 0
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
    val initAcceptanceCount = 0.0
    val initTuningPar = 0.0

    val fullStateInit = FullState(initAlphaCoefs, initBetaCoefs, initZetaCoefs, initGammas, initLambdas, initTauHS, initmt, inittaus, initAcceptanceCount, initTuningPar, initAcceptanceCount, initTuningPar)
    calculateAllStates(info.noOfIter, info, fullStateInit)
  }

  /**
   * Function for updating indicators, interactions and final interaction coefficients
   */
  override def nextIndicsInters(oldfullState: FullState, info: InitialInfo, inBurnIn: Boolean): FullState = {

    val curGammaEstim = (DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels))
    val curLambdaEstim = (DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels))
    var lsiLambda = 0.0
    var lsiTauHS = 0.0
    var curTauHS = 0.0
    val njk = info.noOfInters //no of interactions
    val batchSize = 50 //batch size for automatic tuning

    var acceptedCountLambda = oldfullState.lambdaCount //counter for the times the proposed value is accepted
    var acceptedCountTauHS = oldfullState.tauHSCount //counter for the times the proposed value is accepted
    iterationCount += 1

    // Update tauHS here because it is common for all gammas
    val oldTauHS = oldfullState.tauHS

    //val stepSizeTauHS = 0.014
    //val stepSizeLambda = 2.5 //sigma

    // Automatic adaptation of the tuning parameters based on paper: http://probability.ca/jeff/ftpdir/adaptex.pdf
    val (stepSizeTauHS, stepSizeLambda) = if (inBurnIn){
      if(iterationCount % batchSize == 0){
        val n = iterationCount / batchSize
        val deltan = scala.math.min(0.01, 1/sqrt(n))

        val accFracHS = acceptedCountTauHS / batchSize
        val accFracLambda = acceptedCountLambda / (batchSize * njk)

        acceptedCountTauHS = 0
        acceptedCountLambda = 0

        if(accFracHS > 0.44){
          lsiTauHS = oldfullState.tauHSTuningPar + deltan
        }else{
          lsiTauHS = oldfullState.tauHSTuningPar - deltan
        }
        if(accFracLambda > 0.44){
          lsiLambda = oldfullState.lambdaTuningPar + deltan
        }else{
          lsiLambda = oldfullState.lambdaTuningPar - deltan
        }
        (exp(lsiTauHS), exp(lsiLambda))
      } else{ // if not batch of 50 completed
        lsiTauHS = oldfullState.tauHSTuningPar
        lsiLambda = oldfullState.lambdaTuningPar
        (exp(lsiTauHS), exp(lsiLambda))
      }
    }else{ // if not in burn-in
      lsiTauHS = oldfullState.tauHSTuningPar
      lsiLambda = oldfullState.lambdaTuningPar
      (exp(lsiTauHS), exp(lsiLambda))
    }

    //    println(s"stepSizeTauHS", stepSizeTauHS)
    //    println(s"stepSizeLambda", stepSizeLambda)

    // 1. Use the proposal N(prevTauHS, stepSize) to propose a new location tauHS* (if value sampled <0 propose again until >0)
    val tauHSStar = breeze.stats.distributions.Gaussian(oldTauHS, stepSizeTauHS).draw()

    // Reject tauHSStar if it is < 0. Based on: https://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
    if(tauHSStar < 0){
      curTauHS = oldTauHS
    }
    else {
      val oldTauHSSQR = scala.math.pow(oldTauHS, 2)
      val tauHSStarSQR = scala.math.pow(tauHSStar, 2)

      /**
       * Function for estimating the sqr of two matrices, do elementwise division if the denominator is not 0 and calculate the sum
       */
      def elementwiseDivisionSQRSum(a: DenseMatrix[Double], b: DenseMatrix[Double]): Double = {
        var sum = 0.0
        val aSQR = a.map(x => scala.math.pow(x, 2))
        val bSQR = b.map(x => scala.math.pow(x, 2))
        for (i <- 0 until info.zetaLevels){
          for (j <- 0 until info.zetaLevels){
            if(bSQR(i,j) != 0){
              sum += aSQR(i,j) / bSQR(i,j)
            }
          }
        }
        sum
      }

      //2. Find the acceptance ratio A. Using the log is better and less prone to errors due to overflows.
      val A = log(oldTauHSSQR + 1) + njk * log(oldTauHS) - log(tauHSStarSQR + 1) - njk * log(tauHSStar) + 0.5 * elementwiseDivisionSQRSum(oldfullState.gammaCoefs, oldfullState.lambdas) * ((1/oldTauHSSQR) - (1/tauHSStarSQR))

      //3. Compare A with a random number from uniform, then accept/reject and store to curLambdaEstim accordingly
      val u = log(breeze.stats.distributions.Uniform(0, 1).draw())

      if(A > u){
        curTauHS = tauHSStar
        if(inBurnIn){
          acceptedCountTauHS += 1
        }
      } else{
        curTauHS = oldTauHS
      }
    }

    info.structureSorted.foreach(item => {
      val j = item.a
      val k = item.b

      // Update lambda_jk
      // 1. Use the proposal N(prevLambda, stepSize) to propose a new location lambda* (if value sampled <0 propose again until >0)
      val oldLambda = oldfullState.lambdas(item.a, item.b)
      val curGamma = oldfullState.gammaCoefs(item.a, item.b)

      val lambdaStar = breeze.stats.distributions.Gaussian(oldLambda, stepSizeLambda).draw()

      // Reject lambdaStar if it is < 0. Based on: https://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
      if(lambdaStar < 0){
        curLambdaEstim(item.a, item.b) = oldLambda
      }
      else {
        val oldLambdaSQR = scala.math.pow(oldLambda, 2)
        val lambdaStarSQR = scala.math.pow(lambdaStar, 2)
        val tauHSSQR = scala.math.pow(curTauHS, 2)

        //2. Find the acceptance ratio A. Using the log is better and less prone to errors.
        val A = log(oldLambdaSQR + 1) + log(oldLambda) - log(lambdaStarSQR) - log(lambdaStar) + (scala.math.pow(curGamma, 2)/(2.0 * tauHSSQR)) * ((1/oldLambdaSQR) - (1/lambdaStarSQR))

        //3. Compare A with a random number from uniform, then accept/reject and store to curLambdaEstim accordingly
        val u = log(breeze.stats.distributions.Uniform(0, 1).draw())

        if(A > u){
          curLambdaEstim(item.a, item.b) = lambdaStar
          if(inBurnIn){
            acceptedCountLambda += 1
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
          SXjkkj - Njkkj * oldfullState.mt(0) - NoOfajForbk * (returnIfExists(oldfullState.zcoefs, item.a) + returnIfExists(oldfullState.zcoefs, item.b))
        } else {
          SXjkkj - Njkkj * oldfullState.mt(0) - NoOfajForbk * (returnIfExists(oldfullState.zcoefs, item.a) + returnIfExists(oldfullState.zcoefs, item.b)) - NoOfakForbj * (returnIfExists(oldfullState.zcoefs, item.b) + returnIfExists(oldfullState.zcoefs, item.a))
        }

      val tauGammajk = 1.0 / scala.math.pow(curLambdaEstim(item.a, item.b) * curTauHS, 2)

      val varPInter = 1.0 / (tauGammajk + oldfullState.mt(1) * Njkkj) //the variance for gammajk
      val meanPInter = (info.gammaPriorMean * tauGammajk + oldfullState.mt(1) * SigmaTheta) * varPInter
      curGammaEstim(item.a, item.b) = breeze.stats.distributions.Gaussian(meanPInter, sqrt(varPInter)).draw()
      curGammaEstim(item.b, item.a) = curGammaEstim(item.a, item.b)

    })
    oldfullState.copy(gammaCoefs = curGammaEstim, lambdas = curLambdaEstim, tauHS = curTauHS, lambdaCount = acceptedCountLambda, lambdaTuningPar = lsiLambda, tauHSCount = acceptedCountTauHS, tauHSTuningPar = lsiTauHS)
  }

  override def getFilesDirectory(): String = "/home/antonia/ResultsFromCloud/Report/symmetricNov/symmetricBoth"

  override def getInputFilePath(): String = getFilesDirectory.concat("/simulInterSymmetricBoth.csv")

  override def getOutputRuntimeFilePath(): String = getFilesDirectory().concat("/ScalaRuntime1mSymmetricBothHorseshoe.txt")

  override def getOutputFilePath(): String = getFilesDirectory.concat("/symmetricBothScalaRes1mHorseshoe.csv")

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

    pw.append("mu ,tau, tauz, tauHS,")
      .append(lambdaTitles)
      .append(",")
      .append( (1 to info.zetaLevels).map { i => "zeta".concat(i.toString) }.mkString(",") )
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
        .append( fullstate.zcoefs.toArray.map { alpha => alpha.toString }.mkString(",") )
        .append(",")
        .append( fullstate.gammaCoefs.toArray.map { theta => theta.toString }.mkString(",") )
        .append("\n")
    }
    pw.close()
  }

}
