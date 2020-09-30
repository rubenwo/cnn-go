package layer

import (
	"fmt"
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"math"
)

type MaxPoolingLayer struct {
	strides []int
	sizes   []int

	outputTensor maths.Tensor
	inputTensor  maths.Tensor

	maxIndices []int
}

func NewMaxPoolingLayer(strides, sizes, inputDims []int) *MaxPoolingLayer {
	fmt.Println(inputDims)
	maxPool := &MaxPoolingLayer{}
	maxPool.strides = strides
	for len(maxPool.strides) < len(inputDims) {
		maxPool.strides = append(maxPool.strides, 1)
	}

	maxPool.sizes = sizes
	for len(maxPool.sizes) < len(inputDims) {
		maxPool.sizes = append(maxPool.sizes, 1)
	}

	maxPool.inputTensor = *maths.NewTensor(inputDims, nil)

	outputDims := make([]int, len(inputDims))
	for i := 0; i < len(outputDims); i++ {
		outputDims[i] = int(math.Ceil((float64(inputDims[i]) - float64(maxPool.sizes[i]) + 1.0) / float64(maxPool.strides[i])))
	}
	maxPool.outputTensor = *maths.NewTensor(outputDims, nil)
	maxPool.maxIndices = make([]int, len(maxPool.outputTensor.Values()))

	return maxPool
}

func (m *MaxPoolingLayer) ForwardPropagation(input maths.Tensor) maths.Tensor {

	fmt.Println(input.Dimensions())
	for i := maths.NewRegionsIteratorWithStrides(&input, m.sizes, []int{}, m.strides); i.HasNext(); {
		nextRegion := i.Next()
		maxIndex := nextRegion.MaxValueIndex()

		m.maxIndices[i.CoordIterator.GetCurrentCount()-1] = maxIndex
		m.outputTensor.SetValue(i.CoordIterator.GetCurrentCount()-1, nextRegion.Values()[maxIndex])
	}

	return m.outputTensor
}
func (m *MaxPoolingLayer) BackwardPropagation(gradient maths.Tensor, lr float64) maths.Tensor {
	inputGradients := m.inputTensor.Zeroes()

	for i := maths.NewRegionsIteratorWithStrides(inputGradients, m.sizes, []int{}, m.strides); i.HasNext(); {
		i.Next()
		maxIndex := m.maxIndices[i.CoordIterator.GetCurrentCount()-1]
		maxCoords := maths.HornerToCoords(maxIndex, m.sizes)
		regionStart := i.CoordIterator.GetCurrentCoords()
		coordsOfMax := func(l, r []int) []int {
			ret := make([]int, len(l))
			for i := 0; i < len(ret); i++ {
				ret[i] = l[i] + r[i]
			}
			return ret
		}(regionStart, maxCoords)
		inputGradients.Set(coordsOfMax, gradient.Values()[i.CoordIterator.GetCurrentCount()-1])
	}

	return *inputGradients
}

func (m *MaxPoolingLayer) OutputDims() []int { return m.outputTensor.Dimensions() }
