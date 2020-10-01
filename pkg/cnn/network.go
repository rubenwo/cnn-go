package cnn

import (
	"fmt"
	"github.com/rubenwo/cnn-go/pkg/cnn/layer"
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"github.com/rubenwo/cnn-go/pkg/cnn/metrics"
	"math/rand"
)

type Network struct {
	layers       []layer.Layer
	inputDims    []int
	learningRate float64
	loss         metrics.LossFunction
}

func New(inputDims []int, learningRate float64, loss metrics.LossFunction) *Network {
	return &Network{
		inputDims:    inputDims,
		learningRate: learningRate,
		layers:       []layer.Layer{},
		loss:         loss}
}

func (n *Network) SetLearningRate(rate float64) {
	n.learningRate = rate
}

func (n *Network) LearningRate() float64 { return n.learningRate }

func (n *Network) AddConvolutionLayer(filterDimensions []int, filterCount int) *Network {
	var dims []int
	if len(n.layers) == 0 {
		dims = n.inputDims
	} else {
		dims = n.layers[len(n.layers)-1].OutputDims()
	}
	n.layers = append(n.layers, layer.NewConvolutionLayer(filterDimensions, filterCount, dims))
	return n
}

func (n *Network) AddMaxPoolingLayer(stride int, dimensions []int) *Network {
	var dims []int
	if len(n.layers) == 0 {
		dims = n.inputDims
	} else {
		dims = n.layers[len(n.layers)-1].OutputDims()
	}
	strides := make([]int, len(dimensions))
	for i := 0; i < len(strides); i++ {
		strides[i] = stride
	}

	n.layers = append(n.layers, layer.NewMaxPoolingLayer(strides, dimensions, dims))
	return n
}

func (n *Network) AddDenseLayer(outputLength int) *Network {
	var dims []int
	if len(n.layers) == 0 {
		dims = n.inputDims
	} else {
		dims = n.layers[len(n.layers)-1].OutputDims()
	}
	n.layers = append(n.layers, layer.NewDenseLayer(outputLength, dims))
	return n
}

func (n *Network) AddOutputLayer() *Network {
	var dims []int
	if len(n.layers) == 0 {
		dims = n.inputDims
	} else {
		dims = n.layers[len(n.layers)-1].OutputDims()
	}

	n.layers = append(n.layers, layer.NewOutputLayer(dims))
	return n
}

// Fit will train the CNN. inputs are the inputs, labels are the labels.
// epochs are the amount of times the network is fitted
// batchSize is the size of every propagation batch.
// if verbose then logging is enabled and is written to to stdout with fmt
// every 'logRate' of iterations a message is written when verbose == true
// onBatchDone is a callback that is called every time a batch is done. This can be used to reduce the learning rate for example
func (n *Network) Fit(inputs []maths.Tensor, labels []maths.Tensor, epochs int, batchSize int, verbose bool, logRate int, onEpochDone func()) {
	fmt.Println("Fit: ignoring batch size")
	if len(labels) != len(inputs) {
		panic("length of labels is not equal to length of inputs")
	}

	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("Starting epoch: %d\n", epoch)
		rand.Shuffle(len(inputs), func(i, j int) {
			inputs[i], inputs[j] = inputs[j], inputs[i]
			labels[i], labels[j] = labels[j], labels[i]
		})
		averageLoss := 0.0
		accuracy := 0.0
		for i := range inputs {
			// train the network
			output := n.forward(inputs[i])
			n.backward(n.loss.CalculateLossDerivative(labels[i].Values(), output.Values()))

			if verbose {
				loss := n.loss.CalculateLoss(labels[i].Values(), output.Values())
				averageLoss += maths.SumFloat64Slice(loss.Values())
				if maths.FindMaxIndexFloat64Slice(labels[i].Values()) == maths.FindMaxIndexFloat64Slice(output.Values()) {
					accuracy++
				}
				if i%logRate == 0 {
					fmt.Printf("Input: %d / %d, average loss for last %d iterations was %f\n", i, len(inputs), logRate, averageLoss/float64(logRate))
					fmt.Printf("Accuracy for the last %d iterations was %.2f\n", logRate, accuracy/float64(logRate))
					fmt.Printf("Using learning rate of %f\n", n.learningRate)
					averageLoss = 0
					accuracy = 0
				}
			}
		}
		fmt.Printf("Completed epoch: %d\n", epoch)
		if onEpochDone != nil {
			onEpochDone()
		}
	}
}

func (n *Network) Predict(input maths.Tensor) []float64 {
	output := n.forward(input)
	return output.Values()
}

func (n *Network) forward(input maths.Tensor) maths.Tensor {
	output := input

	for _, l := range n.layers {
		output = l.ForwardPropagation(output)
	}

	return output
}

func (n *Network) backward(outputGradient maths.Tensor) maths.Tensor {
	inputGradient := outputGradient

	for i := len(n.layers) - 1; i >= 0; i-- {
		inputGradient = n.layers[i].BackwardPropagation(inputGradient, n.learningRate)

	}
	return inputGradient
}
