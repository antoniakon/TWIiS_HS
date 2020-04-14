package mcmc.gibbs

import java.io._
import java.util.concurrent.Executors

abstract class VariableSelection {
  def getFilesDirectory() : String
  def getInputFilePath(): String
  def getOutputRuntimeFilePath() : String
  def getOutputFilePath() : String
  def variableSelection(info: InitialInfo)
  protected def nextmutau(oldfullState: FullState, info: InitialInfo): FullState
  protected def nexttaus(oldfullState: FullState, info: InitialInfo):FullState
  protected def nextCoefs(oldfullState: FullState, info: InitialInfo):FullState
  protected def nextIndicsInters(oldfullState: FullState, info: InitialInfo):FullState
  private val executor = Executors.newSingleThreadExecutor()

  protected final def calculateAllStates(n:Int, info: InitialInfo, fstate:FullState) = {
    //with recursion
//    calculateNewState(n, info, fstate, FullStateList(List(fstate)))
    val njk = info.structure.sizeOfStructure() // Number of levels of interactions
    printTitlesToFile(info)

    //Burn-in period
    val burnInStates = calculateNewState(info.burnIn, info, fstate, FullStateList(Vector()))

    val writeBufferSize = 1000
    val wantedIterations = writeBufferSize * info.thin

    var remainingIterations = n

    //var lastState = fstate //Before burn-in period addition
    var lastState = burnInStates.fstateL.last
    while (remainingIterations > 0) {

      val iterations = if (remainingIterations >= wantedIterations) {
        wantedIterations
      } else {
        remainingIterations
      }
      remainingIterations -= wantedIterations

      val toWrite = calculateNewState(iterations, info, lastState, FullStateList(Vector()))
      lastState = toWrite.fstateL.last
      //now write this buffer
      executor.execute { () => printToFile(toWrite) }

      if(remainingIterations < 0){
        val ar = lastState.count/(njk * (info.noOfIter + info.burnIn))
        println("acceptance rate")
        println(ar)

      }
    }
    executor.shutdown()
  }

  protected def printTitlesToFile(initialInfo: InitialInfo): Unit

  protected def printToFile(fullStateList: FullStateList): Unit

  @annotation.tailrec
  private final def calculateNewState(n:Int, info: InitialInfo, fstate:FullState, fstateList:FullStateList): FullStateList = {
    //println(fstate.acoefs)
    if (n==0) fstateList
    else{
      println(n)
      val latestFullyUpdatedState: FullState = calculateNextState(info, fstate)
      if((n % info.thin).equals(0)) {
        calculateNewState(n-1, info, latestFullyUpdatedState, FullStateList(fstateList.fstateL :+ latestFullyUpdatedState))
      }
      else calculateNewState(n-1, info, latestFullyUpdatedState, fstateList)
    }
  }

  private def calculateNextState(info: InitialInfo, fstate: FullState): FullState = {
    val latestmt = nextmutau(fstate, info)
    val latesttaus = nexttaus(latestmt, info)
    val latestcoefs = nextCoefs(latesttaus, info)
    val latestFullyUpdatedState = nextIndicsInters(latestcoefs, info)
    latestFullyUpdatedState
  }

  // Calculation of the execution time
  final def time[A](f: => A): A = {
    val s = System.nanoTime
    val ret = f
    val execTime = (System.nanoTime - s) / 1e6
    println("time: " + execTime + "ms")
    val bw = new BufferedWriter(new FileWriter(new File(getOutputRuntimeFilePath())))
    bw.write(execTime.toString)
    bw.close()
    ret
  }

}
