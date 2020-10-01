package main

import (
	"fmt"
	"github.com/rubenwo/cnn-go/pkg/cnn"
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"github.com/rubenwo/cnn-go/pkg/cnn/metrics"
	"github.com/rubenwo/cnn-go/pkg/mnist"
	"log"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	labels, err := mnist.ReadLabels("./assets/mnist/train-labels-idx1-ubyte", 10000)
	if err != nil {
		log.Fatal(err)
	}
	labelTensors := mnist.LabelsToTensors(labels)
	imageTensors, err := mnist.ReadGrayImages("./assets/mnist/train-images-idx3-ubyte", 10000)
	if err != nil {
		log.Fatal(err)
	}

	nn := cnn.New([]int{28, 28}, 0.005, &metrics.CrossEntropyLoss{})

	nn.AddConvolutionLayer([]int{3, 3}, 8).
		AddMaxPoolingLayer(2, []int{2, 2}).
		AddDenseLayer(10). // 0-9
		AddOutputLayer()

	nn.Fit(imageTensors, labelTensors, 12, 64, true, 100, func() {
		nn.SetLearningRate(nn.LearningRate() * 0.82)
	})

	img := maths.NewTensor([]int{28, 28}, nil)
	img.Randomize()

	prediction := nn.Predict(*img)
	fmt.Println(prediction)
}
