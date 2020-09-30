package cnn

import (
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

func (n *Network) Fit(inputs []maths.Tensor, labels []maths.Tensor, epochs int, batchSize int, verbose bool) {

}

func (n *Network) Predict(input maths.Tensor) []float64 {
	return nil
}
