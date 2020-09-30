package layer

import (
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
)

type DenseLayer struct {
	weights maths.Tensor
	biases  []float64

	inputDims  []int
	outputDims []int

	recentOutput []float64
	recentInput  maths.Tensor
}

func NewDenseLayer(outputLength int, inputDims []int) *DenseLayer {
	dense := &DenseLayer{}
	dense.inputDims = inputDims
	dense.recentInput = *maths.NewTensor(inputDims, nil)
	dense.recentOutput = make([]float64, outputLength)
	dense.outputDims = []int{outputLength}

	dense.weights = *maths.NewTensor(append(inputDims, outputLength), nil)
	dense.weights.Randomize()

	dense.biases = make([]float64, outputLength)

	return dense
}

func (d *DenseLayer) ForwardPropagation(input maths.Tensor) maths.Tensor {
	d.recentInput = input

	i := maths.NewRegionsIterator(&d.weights, d.inputDims, []int{})
	for i.HasNext() {
		d.recentOutput[i.CoordIterator.GetCurrentCount()] = i.Next().InnerProduct(&input)
	}
	d.recentOutput = func(l, r []float64) []float64 {
		ret := make([]float64, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] + r[i]
		}
		return ret
	}(d.recentOutput, d.biases)

	return *maths.NewTensor([]int{len(d.recentOutput)}, d.recentOutput)
}

func (d *DenseLayer) BackwardPropagation(gradient maths.Tensor, lr float64) maths.Tensor {
	var weightsGradient *maths.Tensor
	for i := 0; i < len(gradient.Values()); i++ {
		newGrads := d.recentInput.MulScalar(gradient.Values()[i])
		if weightsGradient == nil {
			weightsGradient = newGrads
		} else {
			weightsGradient = weightsGradient.AppendTensor(newGrads, len(d.weights.Dimensions()))
		}
	}

	inputGradient := maths.NewTensor(d.inputDims, nil)
	j := maths.NewRegionsIterator(&d.weights, d.inputDims, []int{})
	for j.HasNext() {
		newGrads := j.Next().MulScalar(gradient.Values()[j.CoordIterator.GetCurrentCount()-1])
		inputGradient = inputGradient.Add(newGrads, 1)
	}

	d.weights = *d.weights.Add(weightsGradient, -1.0*lr)

	oValMult := func(l []float64, r float64) []float64 {
		ret := make([]float64, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] * r
		}
		return ret
	}(gradient.Values(), -1.0*lr)

	d.biases = func(l, r []float64) []float64 {
		ret := make([]float64, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] + r[i]
		}
		return ret
	}(d.biases, oValMult)

	return *inputGradient
}

func (d *DenseLayer) OutputDims() []int { return d.outputDims }
