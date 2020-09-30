package cnn

import (
	"fmt"
	"github.com/rubenwo/cnn-go/pkg/cnn/layer"
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"github.com/rubenwo/cnn-go/pkg/cnn/metrics"
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

	for epoch := 0; epoch < epochs; epoch++ {
		if len(labels) != len(inputs) {
			panic("length of labels is not equal to length of inputs")
		}
		propagation := 0
		for i := range inputs {
			output := n.forward(inputs[i])

			label := labels[i]
			output = n.backward(n.loss.CalculateLossDerivative(label.Values(), output.Values()))
			propagation++
			// TODO: Add logging if verbose is true
			if verbose && propagation >= logRate {
				fmt.Println("TODO: Add logging")
				propagation = 0
			}
		}
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
