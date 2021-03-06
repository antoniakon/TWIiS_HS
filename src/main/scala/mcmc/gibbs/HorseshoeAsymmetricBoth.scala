package mcmc.gibbs

import java.io.{File, FileWriter, PrintWriter}

import breeze.linalg.{*, DenseMatrix, DenseVector, max, sum}
import breeze.numerics.{erf, exp, pow, sqrt}
import breeze.stats.distributions.LogNormal
import spire.random.Gaussian
import structure.DVStructure

import scala.math.log

/**
 * Variable selection with Horseshoe. Implementation for asymmetric main effects and asymmetric interactions.
 * Two main effects αs and βs, and their interactions thetas
 * Model: X_ijk | mu,a_j,b_k , tauHS, lambda_jk, theta_jk, tau  ~ N(mu + a_j + b_k + I_jk * theta_jk , τ^−1 )
 * Using gamma priors for taua and taub, Cauchy(0,1)+ for tauHS and lambda_jk
 * Variable selection with Horseshoe: theta_jk| tauHS, lambda_jk ~ N(0 , tauHS^2, lambda_jk^2^ )
 * Asymmetric main effects: as and bs come from a different distribution
 * Asymmetric Interactions: theta_jk is different from theta_kj
 **/
class HorseshoeAsymmetricBoth extends VariableSelection {
  final val Pi = java.lang.Math.PI
  private var iterationCount = 0
  override def variableSelection(info: InitialInfo) = {
    // Initialise case class objects
    val initmt = DenseVector[Double](0.0, 1.0)
    val inittaus = DenseVector[Double](1.0, 1.0)
    val initAlphaCoefs = DenseVector.zeros[Double](info.alphaLevels)
    val initBetaCoefs = DenseVector.zeros[Double](info.betaLevels)
    val initZetaCoefs = DenseVector.zeros[Double](info.zetaLevels)
    val initGammas = DenseMatrix.zeros[Double](info.alphaLevels, info.betaLevels) //Thetas represent the interaction coefficients gamma for this case
    val initLambdas = DenseMatrix.ones[Double](info.alphaLevels, info.betaLevels)
    val initTauHS = 1.0
    val initAcceptanceCountLambda = DenseMatrix.zeros[Double](info.alphaLevels, info.betaLevels)
    val initTuningParLambda = DenseMatrix.zeros[Double](info.alphaLevels, info.betaLevels)
    val initAcceptanceCountTau = 0.0
    val initTuningParTau = 0.0

    val fullStateInit = FullState(initAlphaCoefs, initBetaCoefs, initZetaCoefs, initGammas, initLambdas, initTauHS, initmt, inittaus, initAcceptanceCountLambda, initTuningParLambda, initAcceptanceCountTau, initTuningParTau)
    calculateAllStates(info.noOfIter, info, fullStateInit)
  }

  /**
   * Function for updating mu and tau
   */
  override def nextmutau(oldfullState: FullState, info: InitialInfo): FullState = {
    val prevtau = oldfullState.mt(1)
    val varMu = 1.0 / (info.tau0 + info.N * prevtau) //the variance for mu
    val meanMu = (info.mu0 * info.tau0 + prevtau * (info.SumObs - sumAllMainInterEff(info.structure, oldfullState.acoefs, oldfullState.bcoefs, oldfullState.gammaCoefs))) * varMu
    val newmu = breeze.stats.distributions.Gaussian(meanMu, sqrt(varMu)).draw()
    //Use the just updated mu to estimate tau
    val newtau = breeze.stats.distributions.Gamma(info.a + info.N / 2.0, 1.0 / (info.b + 0.5 * YminusMuAndEffects(info.structure, newmu, oldfullState.acoefs, oldfullState.bcoefs, oldfullState.gammaCoefs))).draw() //  !!!!TO SAMPLE FROM THE GAMMA DISTRIBUTION IN BREEZE THE β IS 1/β
    oldfullState.copy(mt = DenseVector(newmu, newtau))
  }

  /**
   * Function for updating taus (taua, taub, tauInt)
   */
  override def nexttaus(oldfullState: FullState, info: InitialInfo): FullState = {

    var sumaj = 0.0
    oldfullState.acoefs.foreachValue(acoef => {
      sumaj += pow(acoef - info.alphaPriorMean, 2)
    })
    sumaj -= (info.alphaLevels - info.alphaLevelsDist) * pow(0 - info.alphaPriorMean, 2) //For the missing effects (if any) added extra in the sum above

    var sumbk = 0.0
    oldfullState.bcoefs.foreachValue(bcoef => {
      sumbk += pow(bcoef - info.betaPriorMean, 2)
    })
    sumbk -= (info.betaLevels - info.betaLevelsDist) * pow(0 - info.betaPriorMean, 2) //For the missing effects (if any) added extra in the sum above

    val newtauAlpha = breeze.stats.distributions.Gamma(info.aPrior + info.alphaLevelsDist / 2.0, 1.0 / (info.bPrior + 0.5 * sumaj)).draw() //sample the precision of alpha from gamma
    val newtauBeta = breeze.stats.distributions.Gamma(info.aPrior + info.betaLevelsDist / 2.0, 1.0 / (info.bPrior + 0.5 * sumbk)).draw() // sample the precision of beta from gamma

    oldfullState.copy(tauab = DenseVector(newtauAlpha, newtauBeta))
  }

  override def nextCoefs(oldfullState: FullState, info: InitialInfo): FullState = {
    val latestAlphaCoefs = nextAlphaCoefs(oldfullState, info)
    nextBetaCoefs(latestAlphaCoefs, info)
  }

  /**
   * Function for updating alpha coefficients
   */
  def nextAlphaCoefs(oldfullState: FullState, info: InitialInfo): FullState = {
    val curAlphaEstim = DenseVector.zeros[Double](info.alphaLevels)

    info.structure.getAllItemsMappedByA().foreach(item => {
      val j = item._1
      val SXalphaj = info.structure.calcAlphaSum(j) // the sum of the observations that have alpha==j
      val Nj = info.structure.calcAlphaLength(j) // the number of the observations that have alpha==j
      val SumBeta = sumBetaEffGivenAlpha(info.structure, j, oldfullState.bcoefs) //the sum of the beta effects given alpha
      val SinterAlpha = sumInterEffGivenAlpha(info.structure, j, oldfullState.gammaCoefs) //the sum of the gamma effects given alpha
      val varPalpha = 1.0 / (oldfullState.tauab(0) + oldfullState.mt(1) * Nj) //the variance for alphaj
      val meanPalpha = (info.alphaPriorMean * oldfullState.tauab(0) + oldfullState.mt(1) * (SXalphaj - Nj * oldfullState.mt(0) - SumBeta - SinterAlpha)) * varPalpha //the mean for alphaj
      curAlphaEstim(j) = breeze.stats.distributions.Gaussian(meanPalpha, sqrt(varPalpha)).draw()
    })
    oldfullState.copy(acoefs = curAlphaEstim)
  }

  /**
   * Function for updating beta coefficients
   */
  def nextBetaCoefs(oldfullState: FullState, info: InitialInfo): FullState = {
    val curBetaEstim = DenseVector.zeros[Double](info.betaLevels)

    info.structure.getAllItemsMappedByB().foreach(item => {
      val k = item._1
      val SXbetak = info.structure.calcBetaSum(k) // the sum of the observations that have beta==k
      val Nk = info.structure.calcBetaLength(k) // the number of the observations that have beta==k
      val SumAlpha = sumAlphaGivenBeta(info.structure, k, oldfullState.acoefs) //the sum of the alpha effects given beta
      val SinterBeta = sumInterEffGivenBeta(info.structure, k, oldfullState.gammaCoefs) //the sum of the gamma/interaction effects given beta
      val varPbeta = 1.0 / (oldfullState.tauab(1) + oldfullState.mt(1) * Nk) //the variance for betak
      val meanPbeta = (info.betaPriorMean * oldfullState.tauab(1) + oldfullState.mt(1) * (SXbetak - Nk * oldfullState.mt(0) - SumAlpha - SinterBeta)) * varPbeta //the mean for betak
      curBetaEstim(k) = breeze.stats.distributions.Gaussian(meanPbeta, sqrt(varPbeta)).draw()
    })
    oldfullState.copy(bcoefs = curBetaEstim)
  }

  /**
   * Function for updating interaction coefficients
   */
  override def nextIndicsInters(oldfullState: FullState, info: InitialInfo, inBurnIn: Boolean): FullState = {
    val curGammaEstim = DenseMatrix.zeros[Double](info.alphaLevels, info.betaLevels)
    val curLambdaEstim = DenseMatrix.zeros[Double](info.alphaLevels, info.betaLevels)
    val njk = info.structure.sizeOfStructure() //no of interactions
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
        acceptedCountLambda = DenseMatrix.zeros[Double](info.alphaLevels, info.betaLevels) //Reset all counts to 0 since batch is completed

        //val sk = breeze.numerics.exp.inPlace(lsiLambda)
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
        // TODO: this breeze.numerics.exp.inPlace(lsiLambda) might give incorrect results bcs it's mutable
        (exp(lsiTauHS), newLamExp, lsiLambda, lsiTauHS)
      }
    }else{ // if not in burn-in
      val lsiTauHS = oldfullState.tauHSTuningPar
      val lsiLambda = oldLambdaTuningParams
      val newLamExp = lsiLambda.map(i=> exp(i))
      // TODO: this breeze.numerics.exp.inPlace(lsiLambda) might give incorrect results bcs it's mutable
      (exp(lsiTauHS), newLamExp, lsiLambda, lsiTauHS)
    }

    // 1. Use the proposal N(prevTauHS, stepSize) to propose a new location tauHS* (if value sampled <0 propose again until >0)
    //    val tauHSStar = breeze.stats.distributions.Gaussian(oldTauHS, stepSizeTauHS).draw()
    val tauHSStar = breeze.stats.distributions.LogNormal(log(oldTauHS), stepSizeTauHS).draw()

    //    val tauHSStar = exp(breeze.stats.distributions.Gaussian(log(oldTauHS), stepSizeTauHS).draw()) //This will always sample positive values.
    // OR
    //    val tauHSStar = oldTauHS * exp(breeze.stats.distributions.LogNormal(0, stepSizeTauHS).draw())

    // Reject tauHSStar if it is < 0. Based on: https://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
    val curTauHS = if(tauHSStar < 0){
      oldTauHS //Return value to curTauHS
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
        for (i <- 0 until info.alphaLevels){
          for (j <- 0 until info.betaLevels){
            if(bSQR(i,j) != 0){
              sum += aSQR(i,j) / bSQR(i,j)
            }
          }
        }
        sum
      }

      //2. Find the acceptance ratio A. Using the log is better and less prone to errors due to underflows.
      // The log implementation is based on: https://umbertopicchini.wordpress.com/2017/12/18/tips-for-coding-a-metropolis-hastings-sampler/
      val JFromTStarToOldT = breeze.stats.distributions.LogNormal(log(tauHSStar), stepSizeTauHS).logPdf(oldTauHS)
      val JFromOldTToTStar = breeze.stats.distributions.LogNormal(log(oldTauHS), stepSizeTauHS).logPdf(tauHSStar)
      val jac = JFromTStarToOldT - JFromOldTToTStar
      // it should either be:
      val A = log(oldTauHSSQR + 1) + njk * log(oldTauHS) - log(tauHSStarSQR + 1) - njk * log(tauHSStar) + 0.5 * elementwiseDivisionSQRSum(oldfullState.gammaCoefs, oldfullState.lambdas) * ((1/oldTauHSSQR) - (1/tauHSStarSQR)) + jac
      // OR
      // val A = log(oldTauHSSQR + 1) + njk * log(oldTauHS) - log(tauHSStarSQR + 1) - njk * log(tauHSStar) + 0.5 * elementwiseDivisionSQRSum(oldfullState.gammaCoefs, oldfullState.lambdas) * ((1/oldTauHSSQR) - (1/tauHSStarSQR)) + log(tauHSStar) - log(oldTauHS)

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

    info.structure.foreach(item => {
      // Update lambda_jk
      // 1. Use the proposal N(prevLambda, stepSize) to propose a new location lambda* (if value sampled <0 propose again until >0)
      val oldLambda = oldfullState.lambdas(item.a, item.b)
      val curGamma = oldfullState.gammaCoefs(item.a, item.b)
      val curStepSizeL = stepSizeLambda(item.a, item.b)

      //val lambdaStar = breeze.stats.distributions.Gaussian(oldLambda, stepSizeLambda(item.a, item.b)).draw()
      val lambdaStar = breeze.stats.distributions.LogNormal(log(oldLambda), stepSizeLambda(item.a, item.b)).draw()
      //val lambdaStar = oldLambda * exp(breeze.stats.distributions.LogNormal(0, curStepSizeL).draw())

      // Reject lambdaStar if it is < 0. Based on: https://darrenjw.wordpress.com/2012/06/04/metropolis-hastings-mcmc-when-the-proposal-and-target-have-differing-support/
      if(lambdaStar < 0){
        curLambdaEstim(item.a, item.b) = oldLambda
      }
      else {
        val oldLambdaSQR = scala.math.pow(oldLambda, 2)
        val lambdaStarSQR = scala.math.pow(lambdaStar, 2)
        val tauHSSQR = scala.math.pow(curTauHS, 2)

        //2. Find the acceptance ratio A. Using the log is better and less prone to errors.
        val JFromLStarToOldL = LogNormal(log(lambdaStar), curStepSizeL).logPdf(oldLambda)
        val JFromOldLToLStar = LogNormal(log(oldLambda), curStepSizeL).logPdf(lambdaStar)
        val jacL = JFromLStarToOldL - JFromOldLToLStar

        val A = (log(oldLambdaSQR + 1) + log(oldLambda) - log(lambdaStarSQR + 1) - log(lambdaStar) + (scala.math.pow(curGamma, 2)/(2.0 * tauHSSQR)) * ((1/oldLambdaSQR) - (1/lambdaStarSQR))) + jacL
        // OR
        //val A = (log(oldLambdaSQR + 1) + log(oldLambda) - log(lambdaStarSQR + 1) - log(lambdaStar) + (scala.math.pow(curGamma, 2)/(2.0 * tauHSSQR)) * ((1/oldLambdaSQR) - (1/lambdaStarSQR))) + log(lambdaStar) - log(oldLambda)

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

      // update gamma_jk
      val Njk = item.list.length // the number of the observations that have alpha==j and beta==k
      val SXjk = item.list.sum // the sum of the observations that have alpha==j and beta==k
      val tauGammajk = 1.0 / scala.math.pow(curLambdaEstim(item.a, item.b) * curTauHS, 2)
      val varInter = 1.0 / (tauGammajk + oldfullState.mt(1) * Njk) //the variance for gammajk
      val meanPInter = (info.gammaPriorMean * tauGammajk + oldfullState.mt(1) * (SXjk - Njk * (oldfullState.mt(0) + oldfullState.acoefs(item.a) + oldfullState.bcoefs(item.b)))) * varInter
      curGammaEstim(item.a, item.b) = breeze.stats.distributions.Gaussian(meanPInter, sqrt(varInter)).draw()
    })

    oldfullState.copy(gammaCoefs = curGammaEstim, lambdas = curLambdaEstim, tauHS = curTauHS, lambdaCount = acceptedCountLambda, lambdaTuningPar = lsiLambda, tauHSCount = acceptedCountTauHS, tauHSTuningPar = lsiTauHS)
  }

  /**
   * Add all the beta effects for a given alpha.
   */
  def sumBetaEffGivenAlpha(structure: DVStructure, alphaIndex: Int, betaEff: DenseVector[Double]): Double = {
    var sum = 0.0
    structure.getAllItemsForGivenA(alphaIndex).foreach(item => {
      sum += item.list.length * betaEff(item.b)
    })
    sum
  }

  /**
   * Add all the alpha effects for a given beta.
   */
  def sumAlphaGivenBeta(structure: DVStructure, betaIndex: Int, alphaEff: DenseVector[Double]): Double = {
    var sum = 0.0
    structure.getAllItemsForGivenB(betaIndex).foreach(item => {
      sum += item.list.length * alphaEff(item.a)
    })
    sum
  }

  /**
   * Calculate the sum of all the alpha and all the beta effects for all the observations.
   */
  def sumAllMainInterEff(structure: DVStructure, alphaEff: DenseVector[Double], betaEff: DenseVector[Double], interEff: DenseMatrix[Double]): Double = {
    var totalsum = 0.0
    structure.foreach(item => {
      totalsum += item.list.length * (alphaEff(item.a) + betaEff(item.b) + interEff(item.a, item.b))
    })
    totalsum
  }

  /**
   * Add all the interaction effects for a given alpha.
   */
  def sumInterEffGivenAlpha(structure: DVStructure, alphaIndex: Int, interEff: DenseMatrix[Double]): Double = {
    var sum = 0.0
    structure.getAllItemsForGivenA(alphaIndex).foreach(item => {
      sum += item.list.length * interEff(item.a, item.b)
    })
    sum
  }

  /**
   * Add all the interaction effects for a given beta.
   */
  def sumInterEffGivenBeta(structure: DVStructure, betaIndex: Int, interEff: DenseMatrix[Double]): Double = {
    var sum = 0.0
    structure.getAllItemsForGivenB(betaIndex).foreach(item => {
      sum += item.list.length * interEff(item.a, item.b)
    })
    sum
  }

  /**
   * Calculate the Yi-mu-u_eff-n_eff- inter_effe. To be used in estimating tau
   */
  def YminusMuAndEffects(structure: DVStructure, mu: Double, alphaEff: DenseVector[Double], betaEff: DenseVector[Double], interEff: DenseMatrix[Double]): Double = {
    var sum = 0.0

    structure.foreach(item => {
      val a = item.a
      val b = item.b
      sum += item.list.foldLeft(0.0)((sum, x) => sum + scala.math.pow(x - mu - alphaEff(a) - betaEff(b) - interEff(a, b) , 2))
    })
    sum
  }

  override def getFilesDirectory(): String = "./data/15x20"

  override def getInputFilePath(): String = getFilesDirectory.concat("/simulInterAsymmetricBoth.csv")

  override def getOutputRuntimeFilePath(): String = getFilesDirectory().concat("/ScalaAsymBothHorseshoe10x15Runtime.txt")

  override def getOutputFilePath(): String = getFilesDirectory.concat("/ScalaAsymBothHorseshoe10x15Results.csv")

  override def printTitlesToFile(info: InitialInfo): Unit = {
    val pw = new PrintWriter(new File(getOutputFilePath))

    val thetaTitles = (1 to info.betaLevels)
      .map { j => "-".concat(j.toString) }
      .map { entry =>
        (1 to info.alphaLevels).map { i => "theta".concat(i.toString).concat(entry) }.mkString(",")
      }.mkString(",")

    val lambdaTitles = (1 to info.betaLevels)
      .map { j => "-".concat(j.toString) }
      .map { entry =>
        (1 to info.alphaLevels).map { i => "lambda".concat(i.toString).concat(entry) }.mkString(",")
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
