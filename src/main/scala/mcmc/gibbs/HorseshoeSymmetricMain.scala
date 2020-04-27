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
  var iterationCount = 0
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
    val meanMu = (info.mu0 * info.tau0 + oldfullState.mt(1) * (info.SumObs - sumAllMainInterEff(info.structure, oldfullState.zcoefs, info.zetaLevels, oldfullState.thcoefs, oldfullState.indics))) * varMu
    val newmu = breeze.stats.distributions.Gaussian(meanMu, sqrt(varMu)).draw()
    //Use the just updated mu to estimate tau
    val newtau = breeze.stats.distributions.Gamma(info.a + info.N / 2.0, 1.0 / (info.b + 0.5 * YminusMuAndEffects(info.structure, newmu, oldfullState.zcoefs, oldfullState.thcoefs, oldfullState.indics))).draw() //  !!!!TO SAMPLE FROM THE GAMMA DISTRIBUTION IN BREEZE THE β IS 1/β
    oldfullState.copy(mt=DenseVector(newmu,newtau))
  }

  /**
   * Function for updating taus (tauz, tauInt)
   */
  override def nexttaus(oldfullState: FullState, info: InitialInfo):FullState= {

    //todo: check if acoef non set values create an issue
    var sumzj = 0.0

    oldfullState.zcoefs.foreachValue( zcoef => {
      sumzj += pow(zcoef - info.alphaPriorMean, 2)
    })

    //todo: check if thcoef non set values create an issue
    var sumThetajk = 0.0
    oldfullState.thcoefs.foreachValue(thcoef => {
      sumThetajk += pow(thcoef -info.thetaPriorMean, 2) // Sum used in sampling from Gamma distribution for the precision of theta/interacions
    })

    val njk = info.structure.sizeOfStructure() // Number of levels of interactions
    val newtauZeta = breeze.stats.distributions.Gamma(info.aPrior + info.zetaLevels / 2.0, 1.0 / (info.bPrior + 0.5 * sumzj)).draw() //sample the precision of alpha from gamma
    val newtauTheta = breeze.stats.distributions.Gamma(info.aPrior + njk / 2.0, 1.0 / (info.bPrior + 0.5 * sumThetajk)).draw() // sample the precision of the interactions gamma from gamma Distribition

    oldfullState.copy(tauabth = DenseVector(newtauZeta, newtauTheta))
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

    (0 until info.zetaLevels).foreach( item => { //For each existing zeta
      val j = item
      val SXZetaj = info.structure.calcZetaSum(j) // the sum of the observations that have zeta == j on either side, not both
      val Nj = info.structure.calcZetaLength(j) // the number of the observations that have zeta == j on either side, not both
      val Njj = info.structure.calcDoubleZetaLength(j) // the number of the observations that have zeta == j on both sides
      val SXZetajDouble = info.structure.calcDoubleZetaSum(j) // the sum of the observations that have zeta == j on both sides
      val SumZeta = sumEffectsOfOtherZetas(info.structure, j, curZetaEstim) //the sum of the other zeta effects given zeta, for which the given z is on either side (but not on both sides)
      val SinterZeta = sumInterEffGivenZeta(info.structure, j, oldfullState.thcoefs, oldfullState.indics) //the sum of the gamma/interaction effects given zeta, for which the given z is on either side (but not on both sides)
      val SinterZetaDoubles = sumInterEffDoublesGivenZeta(info.structure, j, oldfullState.thcoefs, oldfullState.indics) //the sum of the gamma/interaction effects given zeta, for which the given z is on both sides
      val varPzeta = 1.0 / (oldfullState.tauabth(0) + oldfullState.mt(1) * Nj + 4 * oldfullState.mt(1) * Njj) //the variance for zetaj
      val meanPzeta = (info.alphaPriorMean * oldfullState.tauabth(0) + oldfullState.mt(1) * (SXZetaj - Nj * oldfullState.mt(0) - SumZeta - SinterZeta + 2 * SXZetajDouble - 2 * Njj * oldfullState.mt(0) - 2 * SinterZetaDoubles )) * varPzeta //the mean for alphaj
      curZetaEstim.update(j, breeze.stats.distributions.Gaussian(meanPzeta, sqrt(varPzeta)).draw())
    })

    oldfullState.copy(zcoefs = curZetaEstim)
  }

  /**
   * Function for updating indicators, interactions and final interaction coefficients
   */
  override def nextIndicsInters(oldfullState: FullState, info: InitialInfo):FullState= {

    val curIndicsEstim = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels)
    val curThetaEstim = DenseMatrix.zeros[Double](info.zetaLevels, info.zetaLevels)
    var count = 0.0

    info.structure.foreach( item => {
      val Njk = item.list.length // the number of the observations that have alpha==j and beta==k
      val SXjk = item.list.sum // the sum of the observations that have alpha==j and beta==k

      val u = breeze.stats.distributions.Uniform(0, 1).draw()

      //log-sum-exp trick
      val thcoef = oldfullState.thcoefs(item.a, item.b)
      val logInitExp = oldfullState.mt(1) * thcoef * (SXjk - Njk * (oldfullState.mt(0) + oldfullState.zcoefs(item.a) + oldfullState.zcoefs(item.b) + 0.5 * thcoef))
      val logProb0 = log(1.0 - info.p) //The log of the probability I=0
      val logProb1 = log(info.p) + logInitExp //The log of the probability I=1
      val maxProb = max(logProb0, logProb1) //Find the max of the two probabilities
      val scaledProb0 = exp(logProb0 - maxProb) //Scaled by subtracting the max value and exponentiating
      val scaledProb1 = exp(logProb1 - maxProb) //Scaled by subtracting the max value and exponentiating
      val newProb0 = scaledProb0 / (scaledProb0 + scaledProb1) //Normalised
      // val newProb1 = scaledProb1 / (scaledProb0 + scaledProb1) //Normalised

      if (newProb0 < u) {
        //prob0: Probability for when the indicator = 0, so if prob0 < u => indicator = 1
        curIndicsEstim(item.a, item.b) = 1.0
        count += 1.0
        val varPInter = 1.0 / (oldfullState.tauabth(1) + oldfullState.mt(1) * Njk) //the variance for gammajk
        val meanPInter = (info.thetaPriorMean * oldfullState.tauabth(1) + oldfullState.mt(1) * (SXjk - Njk * (oldfullState.mt(0) + oldfullState.zcoefs(item.a) + oldfullState.zcoefs(item.b)))) * varPInter
        curThetaEstim(item.a, item.b) = breeze.stats.distributions.Gaussian(meanPInter, sqrt(varPInter)).draw()
      }
      else {
        //Update indicator and current interactions if indicator = 0.0
        curIndicsEstim(item.a,item.b) = 0.0
        curThetaEstim(item.a,item.b) = breeze.stats.distributions.Gaussian(info.thetaPriorMean, sqrt(1 / oldfullState.tauabth(1))).draw() // sample from the prior of interactions
      }
    })

    oldfullState.copy(thcoefs = curThetaEstim, indics = curIndicsEstim, finalCoefs = curThetaEstim*:*curIndicsEstim)

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
  def sumAllMainInterEff(structure: DVStructure, zetaEff: DenseVector[Double], nz: Int, interEff: DenseMatrix[Double], indics: DenseMatrix[Double]): Double = {
    var totalsum = 0.0
    structure.foreach(item => {
      totalsum += item.list.length * (zetaEff(item.a) + zetaEff(item.b) + indics(item.a, item.b) * interEff(item.a, item.b))
    })
    totalsum
  }

  /**
   * Add all the interaction effects for a given zeta. Adds all the interactions for which zeta is on either side. Includes the doubles bcs getZetasItemsForGivenZ uses a structure that includes everything
   */
  def sumInterEffGivenZeta(structure: DVStructure, zetaIndex: Int, interEff: DenseMatrix[Double], indics: DenseMatrix[Double]): Double = {
    structure.getAllOtherZetasItemsForGivenZ(zetaIndex).map(elem => elem._2.length * indics(elem._1._1, elem._1._2) * interEff(elem._1._1, elem._1._2)).reduce(_+_)
  }

  /**
   * Add all the interaction effects for a given zeta which is double (zeta,zeta)
   */
  def sumInterEffDoublesGivenZeta(structure: DVStructure, zetaIndex: Int, interEff: DenseMatrix[Double], indics: DenseMatrix[Double]): Double = {
    structure.getAllDoubleZetasItemsForGivenZ(zetaIndex).map(elem => elem._2.length * indics(elem._1._1, elem._1._2) * interEff(elem._1._1, elem._1._2)).reduce(_+_)
  }

  /**
   * Calculate the Yi-mu-u_eff-n_eff- inter_effe. To be used in estimating tau
   */
  def YminusMuAndEffects(structure:DVStructure, mu: Double, zetaEff: DenseVector[Double], interEff: DenseMatrix[Double], indics: DenseMatrix[Double]): Double = {
    var sum = 0.0

    structure.foreach( item => {
      val a = item.a
      val b = item.b
      sum += item.list.map(x => scala.math.pow(x - mu - zetaEff(a) - zetaEff(b) - interEff(a, b) * indics(a, b), 2)).sum
    })
    sum
  }

  override def getFilesDirectory(): String = "/home/antonia/ResultsFromCloud/Report/symmetricNov/symmetricMain"

  override def getInputFilePath(): String = getFilesDirectory.concat("/simulInterSymmetricMain.csv")

  override def getOutputRuntimeFilePath(): String = getFilesDirectory().concat("/ScalaRuntime10mSymmetricMainTRYYY.txt")

  override def getOutputFilePath(): String = getFilesDirectory.concat("/symmetricMainScalaResTRYYY.csv")

  override def printTitlesToFile(info: InitialInfo): Unit = {
    val pw = new PrintWriter(new File(getOutputFilePath()))

    val thetaTitles = (1 to info.zetaLevels)
      .map { j => "-".concat(j.toString) }
      .map { entry =>
        (1 to info.zetaLevels).map { i => "theta".concat(i.toString).concat(entry) }.mkString(",")
      }.mkString(",")

    val indicsTitles = (1 to info.zetaLevels)
      .map { j => "-".concat(j.toString) }
      .map { entry =>
        (1 to info.zetaLevels).map { i => "indics".concat(i.toString).concat(entry) }.mkString(",")
      }.mkString(",")

    pw.append("mu ,tau, tauz, tauInt,")
      .append( (1 to info.zetaLevels).map { i => "zeta".concat(i.toString) }.mkString(",") )
      .append(",")
      .append(thetaTitles)
      .append(",")
      .append(indicsTitles)
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
        .append( fullstate.tauabth.toArray.map { tau => tau.toString }.mkString(",") )
        .append(",")
        .append( fullstate.zcoefs.toArray.map { alpha => alpha.toString }.mkString(",") )
        .append(",")
        .append( fullstate.finalCoefs.toArray.map { theta => theta.toString }.mkString(",") )
        .append(",")
        .append( fullstate.indics.toArray.map { ind => ind.toString }.mkString(",") )
        .append("\n")
    }
    pw.close()
  }

}
