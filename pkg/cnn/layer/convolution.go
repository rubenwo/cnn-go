package layer

import (
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"math"
)

type ConvolutionLayer struct {
	filters              maths.Tensor
	filterDimensionSizes []int
	ccMapSize            []int
	outputDimensions     []int
	recentInput          maths.Tensor
}

func NewConvolutionLayer(filterDimensionSizes []int, depth int, inputDims []int) *ConvolutionLayer {
	conv := &ConvolutionLayer{}

	conv.filterDimensionSizes = filterDimensionSizes
	for len(conv.filterDimensionSizes) < len(inputDims) {
		conv.filterDimensionSizes = append(conv.filterDimensionSizes, 1)
	}

	conv.ccMapSize = maths.SubtractIntSlices(inputDims, maths.AddIntToAll(conv.filterDimensionSizes, -1))

	conv.filters = *maths.NewTensor(append(conv.filterDimensionSizes, depth), nil)

	randLimits := math.Sqrt(2) / math.Sqrt(float64(maths.ProductIntSlice(inputDims)))
	conv.filters.Randomize()
	conv.filters = *conv.filters.MulScalar(randLimits)

	conv.outputDimensions = append(conv.ccMapSize, depth)

	return conv
}

func (c *ConvolutionLayer) crossCorrelationMap(base, filter *maths.Tensor, ccMapSize, padding []int) *maths.Tensor {
	product := 1
	for _, val := range ccMapSize {
		product *= val
	}

	ccMapValues := make([]float64, product)

	var regions []*maths.ValuesIterator

	rii := maths.NewRegionsIteratorIterator(base, filter.Dimensions(), padding)
	for rii.HasNext() {
		regions = append(regions, rii.Next())
	}

	if len(ccMapValues) != len(regions) {
		panic("len(ccMapValues) != len(regions) in ConvolutionLayer.crossCorrelationMap")
	}

	for i := 0; i < len(ccMapValues); i++ {
		ccMapValues[i] = regions[i].InnerProduct(filter)
	}

	return maths.NewTensor(ccMapSize, ccMapValues)
}

func (c *ConvolutionLayer) ForwardPropagation(input maths.Tensor) maths.Tensor {
	c.recentInput = input
	var output *maths.Tensor

	for i := maths.NewRegionsIterator(&c.filters, c.filterDimensionSizes, []int{}); i.HasNext(); {
		newMap := c.crossCorrelationMap(&input, i.Next(), c.ccMapSize, []int{})
		if output == nil {
			output = newMap
		} else {
			output = output.AppendTensor(newMap, len(c.filters.Dimensions()))
		}
	}

	return *output
}
func (c *ConvolutionLayer) BackwardPropagation(gradient maths.Tensor, lr float64) maths.Tensor {
	var filterGradients *maths.Tensor
	inputGradients := c.recentInput.Zeroes()

	outputGradientSize := gradient.FirstDimsCopy(len(gradient.Dimensions()) - 1)

	i := maths.NewRegionsIterator(&gradient, outputGradientSize, []int{})
	j := maths.NewRegionsIterator(&c.filters, c.filterDimensionSizes, []int{})

	for i.HasNext() && j.HasNext() {
		outputLayer := i.Next()
		filterLayer := j.Next()

		newMap := c.crossCorrelationMap(&c.recentInput, outputLayer, c.filterDimensionSizes, []int{})
		if filterGradients == nil {
			filterGradients = newMap
		} else {
			filterGradients = filterGradients.AppendTensor(newMap, len(gradient.Dimensions()))
		}

		padding := maths.AddIntToAll(filterLayer.Dimensions(), -1)

		flippedFilter := filterLayer.Flip()
		currentInputGradient := c.crossCorrelationMap(outputLayer, flippedFilter, inputGradients.Dimensions(), padding)

		inputGradients = inputGradients.Add(currentInputGradient, 1)
	}

	c.filters = *c.filters.Add(filterGradients, -1*lr)
	return *inputGradients
}

func (c *ConvolutionLayer) OutputDims() []int { return c.outputDimensions }
