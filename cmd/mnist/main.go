package main

import (
	"github.com/rubenwo/cnn-go/pkg/cnn"
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"github.com/rubenwo/cnn-go/pkg/cnn/metrics"
	"github.com/rubenwo/cnn-go/pkg/images"
	"github.com/rubenwo/cnn-go/pkg/mnist"
	"image"
	"log"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	labels, err := mnist.ReadLabels("./assets/mnist/train-labels-idx1-ubyte", 60000)
	if err != nil {
		log.Fatal(err)
	}
	labelTensors := mnist.LabelsToTensors(labels)
	imageTensors, err := mnist.ReadGrayImages("./assets/mnist/train-images-idx3-ubyte", 60000)
	if err != nil {
		log.Fatal(err)
	}

	valLabels, err := mnist.ReadLabels("./assets/mnist/t10k-labels-idx1-ubyte", 10000)
	if err != nil {
		log.Fatal(err)
	}
	valLabelTensors := mnist.LabelsToTensors(valLabels)
	valImageTensors, err := mnist.ReadGrayImages("./assets/mnist/t10k-images-idx3-ubyte", 10000)
	if err != nil {
		log.Fatal(err)
	}

	nn := cnn.New([]int{28, 28}, 0.005, &metrics.CrossEntropyLoss{})

	nn.AddConvolutionLayer([]int{3, 3}, 8).
		AddMaxPoolingLayer(2, []int{2, 2}).
		AddFullyConnectedLayer(10). // 0-9
		AddSoftmaxLayer()

	nn.Fit(imageTensors, labelTensors, valImageTensors, valLabelTensors, 12, 64, true, 100, func() {
		nn.SetLearningRate(nn.LearningRate() * 0.82)
	})

	nn.Validate(valImageTensors, valLabelTensors)

}

func readImages() ([]maths.Tensor, []maths.Tensor) {
	zero, err := images.GrayScaleImageFromPath("./assets/digits/0.png")
	if err != nil {
		log.Fatal(err)
	}
	one, err := images.GrayScaleImageFromPath("./assets/digits/1.png")
	if err != nil {
		log.Fatal(err)
	}
	two, err := images.GrayScaleImageFromPath("./assets/digits/2.png")
	if err != nil {
		log.Fatal(err)
	}
	three, err := images.GrayScaleImageFromPath("./assets/digits/3.png")
	if err != nil {
		log.Fatal(err)
	}
	four, err := images.GrayScaleImageFromPath("./assets/digits/4.png")
	if err != nil {
		log.Fatal(err)
	}
	five, err := images.GrayScaleImageFromPath("./assets/digits/5.png")
	if err != nil {
		log.Fatal(err)
	}
	six, err := images.GrayScaleImageFromPath("./assets/digits/6.png")
	if err != nil {
		log.Fatal(err)
	}
	seven, err := images.GrayScaleImageFromPath("./assets/digits/7.png")
	if err != nil {
		log.Fatal(err)
	}
	eight, err := images.GrayScaleImageFromPath("./assets/digits/8.png")
	if err != nil {
		log.Fatal(err)
	}
	nine, err := images.GrayScaleImageFromPath("./assets/digits/9.png")
	if err != nil {
		log.Fatal(err)
	}
	nums := []*image.Gray{zero, one, two, three, four, five, six, seven, eight, nine}
	imageTensors := make([]maths.Tensor, len(nums))
	for i := 0; i < len(nums); i++ {
		pixels := make([]float64, 28*28)
		imageTensors[i] = *maths.NewTensor([]int{28, 28}, pixels)
	}
	return nil, nil
}
