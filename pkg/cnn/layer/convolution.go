package layer

import (
	"fmt"
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

	addAllFilterDimSizes := func(l []int, r int) []int {
		ret := make([]int, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] + r
		}
		return ret
	}(filterDimensionSizes, -1)

	conv.ccMapSize = func(l, r []int) []int {
		ret := make([]int, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] - r[i]
		}
		return ret
	}(inputDims, addAllFilterDimSizes)

	conv.filters = *maths.NewTensor(append(conv.filterDimensionSizes, depth), nil)

	product := 1
	for i := 0; i < len(inputDims); i++ {
		product *= inputDims[i]
	}
	randLimits := math.Sqrt(2) / math.Sqrt(float64(product))
	conv.filters.Randomize()
	conv.filters = *conv.filters.MulScalar(randLimits)

	conv.outputDimensions = append(conv.ccMapSize, depth)

	return conv
}

func crossCorrelationMap(base, filter maths.Tensor, ccMapSize, padding []int) *maths.Tensor {
	product := 1
	for _, val := range ccMapSize {
		product *= val
	}

	ccMapValues := make([]float64, product)

	var regions []*maths.ValuesIterator

	rii := maths.NewRegionsIteratorIterator(&base, filter.Dimensions(), padding)
	for rii.HasNext() {
		regions = append(regions, rii.Next())
	}

	if len(ccMapValues) != len(regions) {
		fmt.Println("Problem")
	}

	for i := 0; i < len(ccMapValues); i++ {
		ccMapValues[i] = regions[i].InnerProduct(&filter)
	}

	return maths.NewTensor(ccMapSize, ccMapValues)
}

func (c *ConvolutionLayer) ForwardPropagation(input maths.Tensor) maths.Tensor {
	c.recentInput = input
	var output *maths.Tensor

	for i := maths.NewRegionsIterator(&input, c.filterDimensionSizes, []int{}); i.HasNext(); {
		newMap := crossCorrelationMap(input, *i.Next(), c.ccMapSize, []int{})
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

		newMap := crossCorrelationMap(c.recentInput, *outputLayer, c.filterDimensionSizes, []int{})
		if filterGradients == nil {
			filterGradients = newMap
		} else {
			filterGradients = filterGradients.AppendTensor(newMap, len(gradient.Dimensions()))
		}

		padding := func(l []int, r int) []int {
			ret := make([]int, len(l))
			for i := 0; i < len(ret); i++ {
				ret[i] = l[i] + r
			}
			return ret
		}(filterLayer.Dimensions(), -1)

		flippedFilter := filterLayer.Flip()
		currentInputGradient := crossCorrelationMap(*outputLayer, *flippedFilter, inputGradients.Dimensions(), padding)

		inputGradients = inputGradients.Add(currentInputGradient, 1)
	}

	c.filters = *c.filters.Add(filterGradients, -1*lr)
	return *inputGradients
}

func (c *ConvolutionLayer) OutputDims() []int { return c.outputDimensions }
