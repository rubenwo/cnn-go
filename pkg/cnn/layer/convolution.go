package layer

import (
	"fmt"
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"math"
	"runtime"
)

type ConvolutionLayer struct {
	filters              maths.Tensor
	filterDimensionSizes []int
	ccMapSize            []int
	outputDimensions     []int
	recentInput          maths.Tensor

	jobs    chan job
	results chan result
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
	conv.filters = *conv.filters.Randomize()
	conv.filters = *conv.filters.MulScalar(randLimits)

	conv.outputDimensions = append(conv.ccMapSize, depth)

	conv.jobs = make(chan job, 1000)
	conv.results = make(chan result, 1000)

	for i := 0; i < runtime.NumCPU(); i++ {
		go worker(conv.jobs, conv.results)
	}

	return conv
}

type job struct {
	i int
	f func(i int) float64
}

type result struct {
	i int
	v float64
}

func worker(jobs <-chan job, results chan<- result) {
	for j := range jobs {
		results <- result{i: j.i, v: j.f(j.i)}
	}
}

// crossCorrelationMap creates a cross-correlation map based on a filter.
// ccMapSize is passed so we don't have to recalculate it every time.
// base is the input where the filter needs to be applied to.
func (c *ConvolutionLayer) crossCorrelationMap(base, filter *maths.Tensor, ccMapSize, padding []int) *maths.Tensor {
	ccMapValues := make([]float64, maths.ProductIntSlice(ccMapSize))

	// We append all regions to slice so we can process each region in parallel. Which might be faster if the regions
	// are large enough
	var regions []*maths.ValuesIterator
	rii := maths.NewRegionsIteratorIterator(base, filter.Dimensions(), padding)
	for rii.HasNext() {
		regions = append(regions, rii.Next())
	}

	// assert regions is not nil
	if regions == nil {
		panic("regions is nil, something went wrong")
	}

	// length of ccMapValues needs to be the same length of regions, else something went wrong
	if len(ccMapValues) != len(regions) {
		panic(fmt.Sprintf("len(ccMapValues)=%d != len(regions)=%d", len(ccMapValues), len(regions)))
	}

	// For now we process each region sequentially because it takes more time to dispatch the values to workers than it
	// takes to range over the regions and calculate the InnerProduct ourselves
	for i := 0; i < len(ccMapValues); i++ {
		ccMapValues[i] = regions[i].InnerProduct(filter)
	}

	//// process each regions Inner Product in parallel using n = runtime.NumCPU() workers
	//for i := range ccMapValues {
	//	// Send job to worker
	//	c.jobs <- job{i: i, f: func(i int) float64 { return regions[i].InnerProduct(filter) }}
	//}
	//for range ccMapValues {
	//	// receive result and set result in ccMapValues
	//	res := <-c.results
	//	ccMapValues[res.i] = res.v
	//}

	// Return a tensor with the cross-correlation map
	return maths.NewTensor(ccMapSize, ccMapValues)
}

func (c *ConvolutionLayer) ForwardPropagation(input maths.Tensor) maths.Tensor {
	c.recentInput = input // might want to rewrite this because it blocks parallel batches
	var output *maths.Tensor

	//Iterate through each filter, appending result to output tensor.
	for iter := maths.NewRegionsIterator(&c.filters, c.filterDimensionSizes, []int{}); iter.HasNext(); {
		newMap := c.crossCorrelationMap(&input, iter.Next(), c.ccMapSize, []int{})
		if output == nil {
			output = newMap
		} else {
			output = output.AppendTensor(newMap, len(c.filters.Dimensions()))
		}
	}

	if output == nil {
		panic("ConvolutionLayer.ForwardPropagation did not work. output == nil")
	}
	return *output
}

func (c *ConvolutionLayer) BackwardPropagation(gradient maths.Tensor, lr float64) maths.Tensor {
	var filterGradients *maths.Tensor
	inputGradients := c.recentInput.Zeroes()

	// We can think of outputGrad as a series of "gradient filters" which we apply to the recent input.
	// This is the size of each of those filters.
	outputGradientSize := gradient.FirstDimsCopy(len(gradient.Dimensions()) - 1)
	iterGradient := maths.NewRegionsIterator(&gradient, outputGradientSize, []int{})
	iterFilter := maths.NewRegionsIterator(&c.filters, c.filterDimensionSizes, []int{})

	for iterGradient.HasNext() && iterFilter.HasNext() {
		outputLayer := iterGradient.Next()
		filterLayer := iterFilter.Next()

		// Calculate derivation filters
		newMap := c.crossCorrelationMap(&c.recentInput, outputLayer, c.filterDimensionSizes, []int{})
		// Append or assign new map to filter grad tensor
		if filterGradients == nil {
			filterGradients = newMap
		} else {
			filterGradients = filterGradients.AppendTensor(newMap, len(gradient.Dimensions()))
		}

		// Calculate derivation input
		// derivation input = derivation  outputs * flipped filters
		// To return the correct sized tensor, this requires some padding - which happens to be (filter sizes - 1)
		padding := maths.AddIntToAll(filterLayer.Dimensions(), -1)
		flippedFilter := filterLayer.Flip()
		currentInputGradient := c.crossCorrelationMap(outputLayer, flippedFilter, inputGradients.Dimensions(), padding)

		inputGradients = inputGradients.Add(currentInputGradient, 1)
	}

	if filterGradients == nil {
		panic("BackwardPropagation did not work correctly in convolution layer. filterGradients == nil")
	}

	// Gradient descent on filters
	c.filters = *c.filters.Add(filterGradients, -1*lr)

	return *inputGradients
}

func (c *ConvolutionLayer) OutputDims() []int { return c.outputDimensions }
