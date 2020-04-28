package mcmc.gibbs

import java.io.{File, FileWriter, PrintWriter}
import breeze.linalg.{DenseMatrix, DenseVector, max}
import breeze.numerics.{exp, pow, sqrt}
import scala.math.{log}
import structure.DVStructure

/**
 * Variable selection with Horseshoe. Implementation for symmetric main effects and asymmetric interactions.
 * Model: X_ijk | mu,a_j,b_k ,tauHS, lambda_jk, gamma_jk,tau  ~ N(mu + z_j + z_k + gamma_jk , τ^−1 )
 * Using gamma priors for taua and taub, Cauchy(0,1)+ for tauHS and lambda_jk
 * Variable selection with Horseshoe: theta_jk| tauHS, lambda_jk ~ N(0 , tauHS^2, lambda_jk^2^ )
 * Symmetric main effects: zs come from the same distribution
 * Asymmetric Interactions: I_jk * theta_jk != I_kj * theta_kj
 **/
class HorseshoeSymmetricMain extends VariableSelection {
  private var iterationCount = 0
  override def variableSelection(info: InitialInfo) = {
    // Initialise case class objects
    val initmt = DenseVector[Double](0.0,1.0)
    val inittaus = DenseVector[Double](1.0) //Only for tauz now
    val initAlphaCoefs = DenseVector.zeros[Double](info.alphaLevels) //Not used in SymmetricMain implementation
    val initBetaCoefs = DenseVector.zeros[Double](info.betaLevels) //Not used in SymmetricMain implementation
    val initZetaCoefs = DenseVector.zeros[Double](info.zetaLevels)
    val initGammas = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels) //zetaLevels in SymmetricMain implementation
    val initLambdas = DenseMatrix.ones[Double](info.zetaLevels, info.zetaLevels) //zetaLevels in SymmetricMain implementation
    val initTauHS = 1.0
    val initAcceptanceCount = 0.0
    val initTuningPar = 0.0

    val fullStateInit = FullState(initAlphaCoefs, initBetaCoefs, initZetaCoefs, initGammas, initLambdas, initTauHS, initmt, inittaus, initAcceptanceCount, initTuningPar, initAcceptanceCount, initTuningPar)
    calculateAllStates(info.noOfIter, info, fullStateInit)
  }

  /**
   * Function for updating mu and tau
   */
  override def nextmutau(oldfullState: FullState, info: InitialInfo): FullState= {
    val varMu = 1.0 / (info.tau0 + info.N * oldfullState.mt(1)) //the variance for mu
    val meanMu = (info.mu0 * info.tau0 + oldfullState.mt(1) * (info.SumObs - sumAllMainInterEff(info.structure, oldfullState.zcoefs, info.zetaLevels, oldfullState.gammaCoefs))) * varMu
    val newmu = breeze.stats.distributions.Gaussian(meanMu, sqrt(varMu)).draw()
    //Use the just updated mu to estimate tau
    val newtau = breeze.stats.distributions.Gamma(info.a + info.N / 2.0, 1.0 / (info.b + 0.5 * YminusMuAndEffects(info.structure, newmu, oldfullState.zcoefs, oldfullState.gammaCoefs))).draw() //  !!!!TO SAMPLE FROM THE GAMMA DISTRIBUTION IN BREEZE THE β IS 1/β
    oldfullState.copy(mt=DenseVector(newmu,newtau))
  }

  /**
   * Function for updating taus (tauz)
   */
  override def nexttaus(oldfullState: FullState, info: InitialInfo):FullState= {

    //todo: check if acoef non set values create an issue
    var sumzj = 0.0

    oldfullState.zcoefs.foreachValue( zcoef => {
      sumzj += pow(zcoef - info.alphaPriorMean, 2)
    })
    sumzj -= (info.zetaLevels - info.zetaLevelsDist) * pow(0 - info.alphaPriorMean, 2) //For the missing effects (if any) added extra in the sum above

    val newtauZeta = breeze.stats.distributions.Gamma(info.aPrior + info.zetaLevels / 2.0, 1.0 / (info.bPrior + 0.5 * sumzj)).draw() //sample the precision of alpha from gamma

    oldfullState.copy(tauab = DenseVector(newtauZeta))
  }

  override def nextCoefs(oldfullState: FullState, info: InitialInfo): FullState = {
    nextZetaCoefs(oldfullState, info)
  }

  /**
   * Function for updating zeta coefficients.
   * Each zeta depends on the other zs, for which the latest update needs to be used.
   */
  def nextZetaCoefs(oldfullState: FullState, info: InitialInfo):FullState={

    val curZetaEstim = DenseVector.zeros[Double](info.zetaLevels)
    curZetaEstim:= oldfullState.zcoefs

    info.structure.getAllZetas().foreach( item => { //For each existing zeta
      val j = item
      val SXZetaj = info.structure.calcZetaSum(j) // the sum of the observations that have zeta == j on either side, not both
      val Nj = info.structure.calcZetaLength(j) // the number of the observations that have zeta == j on either side, not both
      val Njj = info.structure.calcDoubleZetaLength(j) // the number of the observations that have zeta == j on both sides
      val SXZetajDouble = info.structure.calcDoubleZetaSum(j) // the sum of the observations that have zeta == j on both sides
      val SumZeta = sumEffectsOfOtherZetas(info.structure, j, curZetaEstim) //the sum of the other zeta effects given zeta, for which the given z is on either side (but not on both sides)
      val SinterZeta = sumInterEffGivenZeta(info.structure, j, oldfullState.gammaCoefs) //the sum of the gamma/interaction effects given zeta, for which the given z is on either side (but not on both sides)
      val SinterZetaDoubles = sumInterEffDoublesGivenZeta(info.structure, j, oldfullState.gammaCoefs) //the sum of the gamma/interaction effects given zeta, for which the given z is on both sides
      val varPzeta = 1.0 / (oldfullState.tauab(0) + oldfullState.mt(1) * Nj + 4 * oldfullState.mt(1) * Njj) //the variance for zetaj
      val meanPzeta = (info.alphaPriorMean * oldfullState.tauab(0) + oldfullState.mt(1) * (SXZetaj - Nj * oldfullState.mt(0) - SumZeta - SinterZeta + 2 * SXZetajDouble - 2 * Njj * oldfullState.mt(0) - 2 * SinterZetaDoubles )) * varPzeta //the mean for alphaj
      curZetaEstim.update(j, breeze.stats.distributions.Gaussian(meanPzeta, sqrt(varPzeta)).draw())
    })

    oldfullState.copy(zcoefs = curZetaEstim)
  }

  /**
   * Function for updating indicators, interactions and final interaction coefficients
   */
  override def nextIndicsInters(oldfullState: FullState, info: InitialInfo, inBurnIn: Boolean):FullState= {
    val curGammaEstim = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels)
    val curLambdaEstim = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels)
    var lsiLambda = 0.0
    var lsiTauHS = 0.0
    var curTauHS = 0.0
    val njk = info.structure.sizeOfStructure() //no of interactions
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

    info.structure.foreach( item => {
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

      // update gamma_jk
      val Njk = item.list.length // the number of the observations that have alpha==j and beta==k
      val SXjk = item.list.sum // the sum of the observations that have alpha==j and beta==k

      val tauGammajk = 1.0 / scala.math.pow(curLambdaEstim(item.a, item.b) * curTauHS, 2)
      val varPInter = 1.0 / (tauGammajk + oldfullState.mt(1) * Njk) //the variance for gammajk
      val meanPInter = (info.gammaPriorMean * tauGammajk + oldfullState.mt(1) * (SXjk - Njk * (oldfullState.mt(0) + oldfullState.zcoefs(item.a) + oldfullState.zcoefs(item.b)))) * varPInter
      curGammaEstim(item.a, item.b) = breeze.stats.distributions.Gaussian(meanPInter, sqrt(varPInter)).draw()
    })

    oldfullState.copy(gammaCoefs = curGammaEstim, lambdas = curLambdaEstim, tauHS = curTauHS, lambdaCount = acceptedCountLambda, lambdaTuningPar = lsiLambda, tauHSCount = acceptedCountTauHS, tauHSTuningPar = lsiTauHS)
  }

  /**
   * Add all the zeta effects for all the other zetas for that specific zeta.
   * e.g. updating z1: (1,1),(1,2),(2,1),(1,3),(1,4),(4,1) => Sum the effects for: z2*NoOfObs for that category + z2*NoOfObs for that category + z3*NoOfObs for that category + z4*NoOfObs for that category + z4*NoOfObs for that category
   */
  def sumEffectsOfOtherZetas(structure: DVStructure, zetaIndex: Int, zetaEff: DenseVector[Double]): Double = {
    //returns the element which is not zetaIndex. It doesn't take into account the cases where both sides are zetaIndex because getAllOtherZetasItemsForGivenZ works on a structure that does not involve the (j,j) cases
    def notZeta(k1: Int, k2: Int): Int={
      if(k1!=zetaIndex) k1
      else k2
    }
    structure.getAllOtherZetasItemsForGivenZ(zetaIndex).map(elem => elem._2.length * zetaEff(notZeta(elem._1._1, elem._1._2))).reduce(_+_)
  }

  /**
   * Calculate the sum of all the zeta 1 and all the zeta 2 effects for all the observations.
   */
  def sumAllMainInterEff(structure: DVStructure, zetaEff: DenseVector[Double], nz: Int, interEff: DenseMatrix[Double]): Double = {
    var totalsum = 0.0
    structure.foreach(item => {
      totalsum += item.list.length * (zetaEff(item.a) + zetaEff(item.b) + interEff(item.a, item.b))
    })
    totalsum
  }

  /**
   * Add all the interaction effects for a given zeta. Adds all the interactions for which zeta is on either side. Includes the doubles bcs getZetasItemsForGivenZ uses a structure that includes everything
   */
  def sumInterEffGivenZeta(structure: DVStructure, zetaIndex: Int, interEff: DenseMatrix[Double]): Double = {
    structure.getAllOtherZetasItemsForGivenZ(zetaIndex).map(elem => elem._2.length * interEff(elem._1._1, elem._1._2)).reduce(_+_)
  }

  /**
   * Add all the interaction effects for a given zeta which is double (zeta,zeta)
   */
  def sumInterEffDoublesGivenZeta(structure: DVStructure, zetaIndex: Int, interEff: DenseMatrix[Double]): Double = {
    structure.getAllDoubleZetasItemsForGivenZ(zetaIndex).map(elem => elem._2.length * interEff(elem._1._1, elem._1._2)).reduce(_+_)
  }

  /**
   * Calculate the Yi-mu-u_eff-n_eff- inter_effe. To be used in estimating tau
   */
  def YminusMuAndEffects(structure:DVStructure, mu: Double, zetaEff: DenseVector[Double], interEff: DenseMatrix[Double]): Double = {
    var sum = 0.0

    structure.foreach( item => {
      val a = item.a
      val b = item.b
      sum += item.list.map(x => scala.math.pow(x - mu - zetaEff(a) - zetaEff(b) - interEff(a, b), 2)).sum
    })
    sum
  }

  override def getFilesDirectory(): String = "/home/antonia/ResultsFromCloud/Report/symmetricNov/symmetricMain"

  override def getInputFilePath(): String = getFilesDirectory.concat("/simulInterSymmetricMain.csv")

  override def getOutputRuntimeFilePath(): String = getFilesDirectory().concat("/ScalaRuntime10mSymmetricMainHorseshoe.txt")

  override def getOutputFilePath(): String = getFilesDirectory.concat("/symmetricMainScalaResHorseshoe.csv")

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
