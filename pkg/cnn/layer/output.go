package layer

import (
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"math"
)

type OutputLayer struct {
	outputDims           []int
	previousSoftMaxInput maths.Tensor
}

func NewOutputLayer(inputDims []int) *OutputLayer {
	return &OutputLayer{
		outputDims: inputDims}
}

func (o *OutputLayer) ForwardPropagation(input maths.Tensor) maths.Tensor {
	o.previousSoftMaxInput = input

	output := input.Zeroes()
	expSum := 0.0

	for i := 0; i < len(input.Values()); i++ {
		output.SetValue(i, math.Exp(input.At(i)))
		expSum += output.At(i)
	}

	for i := 0; i < len(output.Values()); i++ {
		output.SetValue(i, output.At(i)/expSum)
	}

	return *output
}
func (o *OutputLayer) BackwardPropagation(gradient maths.Tensor, lr float64) maths.Tensor {
	d := o.derivatives(o.previousSoftMaxInput)
	return *gradient.MulElem(&d)
}

func (o *OutputLayer) OutputDims() []int {
	return o.outputDims
}

func (o *OutputLayer) derivatives(input maths.Tensor) maths.Tensor {
	output := make([]float64, input.Len())
	expSum := 0.0

	for i := 0; i < input.Len(); i++ {
		output[i] = math.Exp(input.At(i))
		expSum += output[i]
	}

	for i := 0; i < input.Len(); i++ {
		output[i] *= expSum - output[i]
	}

	for i := 0; i < input.Len(); i++ {
		output[i] /= expSum * expSum
	}

	return *maths.NewTensor(input.Dimensions(), output)
}
