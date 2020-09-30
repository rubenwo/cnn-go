package main

import (
	"fmt"
	"github.com/rubenwo/cnn-go/pkg/cnn"
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"github.com/rubenwo/cnn-go/pkg/cnn/metrics"
	"github.com/rubenwo/cnn-go/pkg/mnist"
)

func main() {
	labels := mnist.ReadLabels("./assets/mnist/train-labels-idx1-ubyte", 10000)
	labelTensors := mnist.LabelsToTensors(labels)
	imageTensors := mnist.ReadGrayImages("./assets/mnist/train-images-idx3-ubyte", 10000)

	nn := cnn.New([]int{28, 28}, 0.0001, &metrics.CrossEntropyLoss{})

	nn.AddConvolutionLayer([]int{3, 3}, 8).
		AddMaxPoolingLayer(2, []int{2, 2}).
		AddDenseLayer(10). // 0-9
		AddOutputLayer()

	nn.Fit(imageTensors, labelTensors, 10, 64, true)

	img := maths.Tensor{}

	prediction := nn.Predict(img)
	fmt.Println(prediction)
}
