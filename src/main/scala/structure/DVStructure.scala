package structure

trait DVStructure {

  /**
    * Calculates the sum of the response y for a given alpha
    */
  def calcAlphaSum(j: Int): Double

  /**
    * Calculates the sum of the response y for a given zeta, not include the cases where k==j
    */
  def calcZetaSum(j: Int): Double

  /**
    * Calculates the sum of the response y for a given beta
    */
  def calcBetaSum(k: Int): Double

  /**
    * Calculates the sum of the responses y for a given alpha and beta
    */
  def calcAlphaBetaSum(j: Int, k: Int): Double
  /**
    * Calculates the number of the responses y for a given alpha
    */
  def calcAlphaLength(j: Int): Double

  /**
    * Calculates the number of the responses y for a given alpha and beta
    */
  def calcAlphaBetaLength(j: Int, k: Int): Double

  /**
    * Calculates the number of the responses y for a given zeta
    */
  def calcZetaLength(j: Int): Double

  /**
    * Calculates the number of the responses y for a given beta
    */
  def calcBetaLength(k: Int): Double

  def calcDoubleZetaSum(zj: Int): Double

  def calcDoubleZetaLength(zj: Int): Double

  def getAllDoubleZetasItemsForGivenZ(z: Int): Map[(Int,Int),DVList]

  /**
    * Returns the DVList for (alpha,beta)
    */
  def getDVList(j: Int, k: Int): DVList

  def foreach[U](f: DVItem => U): Unit

  def getAllItemsForGivenA(a : Int): List[DVItem]

  def getAllOtherZetasItemsForGivenZ(z: Int): Map[(Int,Int),DVList]

  def getZetasItemsForGivenZ(z: Int): Map[(Int,Int),DVList]

  def getAllItemsForGivenB(b : Int): List[DVItem]

  def getAllItemsMappedByA() : Map[Int, List[DVItem]]

  def getAllItemsMappedByB() : Map[Int, List[DVItem]]

  def sizeOfStructure(): Int

  def sizeOfDouble(): Int


}
