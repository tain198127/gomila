package chapter5

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

/**
sigmod函数，核心参数
*/
func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

/**
随机梯度上升算法
@dataMatrixIn 输入的矩阵
@label 特征矩阵
@trainTimes 训练次数
@diffAccuracy 步长的精确度
*/

func RandomGradAscent(dataMatrixIn [][]float64, label []float64, trainTimes int, diffAccuracy float64) []float64 {
	var m = len(dataMatrixIn)    //行数
	var n = len(dataMatrixIn[0]) //列数
	var x = make([]float64, n)   //造一个空的数组，长度是输入矩阵的列数
	for i := range x {
		x[i] = float64(1)
	} //每个列的值都是1
	for t := 0; t < trainTimes; t++ { //训练次数
		for i := 0; i < m; i++ { //行
			var deltaX = float64(4/(1+t+i)) + diffAccuracy //微小的变化,目的是拉普拉斯平滑
			var rdmIdx = rand.Int31n(int32(m))             //随机数
			//var rdmIdx = i //随机数
			var row = dataMatrixIn[rdmIdx] //随机内容
			var sum float64 = 0
			for k, v := range row {
				//k表示key，v表示value，也就是，k表示index，v表示值
				sum += v * x[k] //这是最难理解的地方，理解了都好办
			} //向量乘机
			//上面的这句话的意思是：随便选一行，让 这行里面的每一列的值 * x[列],然后相加.
			//也就是 z=w0*x0+w1*x1+w2*x2......
			var deltaY = label[rdmIdx] - sigmoid(sum) //目标值得差距

			for k, v := range row {
				x[k] = x[k] + deltaX*deltaY*v
			}
		}
	}
	return x
}

/**
分类
*/
func Classfy(data []float64, train_vec []float64) float64 {
	var sum float64
	for i := 0; i < len(data); i++ {
		sum += data[i] * train_vec[i]
	}

	result := sigmoid(sum)
	if result > 0.5 {
		return 1.0
	}
	return 0.0
}

/**
读取数据转换成向量
*/
func readData(path string) ([][]float64, []float64) {
	file, error := ioutil.ReadFile(path)
	if error != nil {
		panic(error)
		log.Fatal("打开文件路径失败")
	}
	document := string(file)
	array := strings.Split(document, "\n")

	var data = make([][]float64, len(array))
	var label = make([]float64, len(array))
	linenumber := 0

	for i := range array {
		row := strings.Split(array[i], "\t")
		data[linenumber] = make([]float64, len(row)-1)
		tmp := make([]float64, len(row))
		for ridx, v := range row {
			val := strings.Trim(v, " ")
			readNum, _ := strconv.ParseFloat(val, 10)
			tmp[ridx] = readNum
		}
		data[linenumber] = tmp[0 : len(tmp)-1]
		label[linenumber] = tmp[len(tmp)-1]

		linenumber++
	}

	return data, label
}

func Test(trainData [][]float64, trainLabel []float64, testData [][]float64, testLabel []float64, trainTimes int, diffAccuracy float64) float64 {
	train := RandomGradAscent(trainData, trainLabel, trainTimes, diffAccuracy)
	var succCount float64 = 0
	var testCount = float64(len(testData))
	for i := range testData {
		result := Classfy(testData[i], train)
		testval := testLabel[i]
		if result == testval {
			succCount++
		}
	}
	var succRate = succCount / testCount
	return succRate
}

func MultiTest(trainFilePath string, testFilePath string, testTimes int) {

	data, label := readData(trainFilePath)
	test, testLabel := readData(testFilePath)
	var sumSuccRate float64 = 0
	ticker := time.Now()
	for i := 0; i < testTimes; i++ {
		sumSuccRate += Test(data, label, test, testLabel, 5000, 0.0001)
	}
	elapsed := time.Since(ticker)
	fmt.Printf("运行%d 次，平均正确率 %.2f%% , 总共耗时 %d毫秒, 平均每次%.1f 毫秒",
		testTimes, sumSuccRate/float64(testTimes)*100, elapsed.Milliseconds(), float64(elapsed.Milliseconds())/float64(testTimes))

}
