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

func (n *Network) AddConvolutionLayer(dimensions []int, filterCount int) *Network {
	return n
}

func (n *Network) AddMaxPoolingLayer(stride int, dimensions []int) *Network {
	return n
}

func (n *Network) AddDenseLayer(outputLength int) *Network {
	return n
}

func (n *Network) AddOutputLayer() *Network {
	return n
}

func (n *Network) Fit(inputs []maths.Tensor, labels []maths.Tensor, epochs int, batchSize int, verbose bool, onBatchDone func()) {
	fmt.Println("ignoring batch size")
	if len(labels) != len(inputs) {
		panic("length of labels is not equal to length of inputs")
	}
	for i := range inputs {
		output := n.forward(inputs[i])
		label := labels[i]
		output = n.backward(n.loss.CalculateLossDerivative(label.Values(), output.Values()))
	}
}

func (n *Network) Predict(input maths.Tensor) []float64 {
	return nil
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
