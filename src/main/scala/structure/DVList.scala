package structure

import scala.collection.mutable.ListBuffer

/**
  * Created by Antonia Kontaratou.
  * A DVList list contains observations of the response y.
  */
class DVList {
  val list= new ListBuffer[Double]

  def addItem(item: Double)= {
    list += item
  }

  def sum: Double= {
    list.sum
  }

  def length: Double= {
    list.length
  }

  def map(f: Double => Double): ListBuffer[Double]={
    list.map(f)
  }

  def foldLeft(z: Double)(op: (Double, Double) => Double): Double={
    list.foldLeft(z)(op)
  }

  override def toString = {
    list.toString
  }

}
