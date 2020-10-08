package main

import (
	"dane.brown.com/src/dane.brown.com/chapter5"
	"flag"
)

func main() {
	trainFilePath := flag.String("train", "/Users/danebrown/develop/github/gomila/resources/chapter5/horseColicTraining.txt", "训练文件矩阵")
	testFilePath := flag.String("test", "/Users/danebrown/develop/github/gomila/resources/chapter5/horseColicTest.txt", "测试文件路径")
	testTimes := flag.Int("times", 10, "训练次数")
	//
	chapter5.MultiTest(*trainFilePath, *testFilePath, *testTimes)

}
