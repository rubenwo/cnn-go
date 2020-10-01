package maths

import (
	"fmt"
	"math"
	"math/rand"
)

type Tensor struct {
	dimension []int
	values    []float64
}

func NewTensor(dimension []int, values []float64) *Tensor {
	if values == nil {
		//The length of our 1-dimensional values array needs to be equivalent to the product of all dimensions
		p := 1
		for _, d := range dimension {
			p *= d
		}
		return &Tensor{dimension: dimension, values: make([]float64, p)}
	}
	return &Tensor{dimension: dimension, values: values}
}

func (t *Tensor) FirstDimsCopy(length int) []int {
	if length > len(t.dimension) {
		panic("length > Tensor.dimension")
	}
	dims := make([]int, length)
	for i := 0; i < length; i++ {
		dims[i] = t.dimension[i]
	}
	return dims
}

func (t *Tensor) MulElem(other *Tensor) *Tensor {
	if t.Len() != other.Len() {
		panic("dimension mismatch in Tensor.MulElem")
	}
	r := Tensor{dimension: t.dimension, values: make([]float64, t.Len())}

	for i := 0; i < t.Len(); i++ {
		r.values[i] = t.values[i] * other.values[i]
	}
	return &r
}

func (t *Tensor) MulScalar(scalar float64) *Tensor {
	values := make([]float64, t.Len())
	for i := 0; i < len(values); i++ {
		values[i] = t.values[i] * scalar
	}

	return NewTensor(t.dimension, values)
}

func (t *Tensor) Add(other *Tensor, factor float64) *Tensor {
	values := make([]float64, len(t.values))
	for i := 0; i < len(values); i++ {
		values[i] = t.values[i] + other.values[i]*factor
	}
	return NewTensor(t.dimension, values)
}

func (t *Tensor) AppendTensor(other *Tensor, resultRank int) *Tensor {
	newDimSizes := make([]int, resultRank)
	for i := len(t.Dimensions()); i < resultRank; i++ {
		newDimSizes[i] = 1
	}

	if len(other.Dimensions()) >= resultRank {
		newDimSizes[resultRank-1] += other.dimension[len(t.dimension)-1]
	} else {
		newDimSizes[resultRank-1] += 1
	}

	newValues := make([]float64, len(t.values)+len(other.values))
	for i := 0; i < len(t.values); i++ {
		newValues[i] = t.values[i]
	}

	for i := len(t.values); i < len(t.values)+len(other.values); i++ {
		newValues[i] = other.values[i-len(t.values)]
	}

	return NewTensor(newDimSizes, newValues)
}

func (t *Tensor) Apply(fn func(val float64, idx int) float64) {
	for i := 0; i < len(t.values); i++ {
		t.values[i] = fn(t.values[i], i)
	}
}

// Randomize uses a rand.NormFloat64() function
func (t *Tensor) Randomize() {
	for i := 0; i < len(t.values); i++ {
		t.values[i] = rand.NormFloat64()
	}
}

func (t *Tensor) SubTensor(dims []int, offset int) *Tensor {
	tensor := NewTensor(dims, nil)
	product := 1
	for _, dim := range dims {
		product *= dim
	}
	for i := offset; i < product+offset; i++ {
		tensor.values[i-offset] = t.values[i]
	}

	return tensor
}

func (t *Tensor) Region(corner1, corner2 []int) *Tensor {
	newDimSizes := make([]int, len(corner1))
	for i := 0; i < len(newDimSizes); i++ {
		newDimSizes[i] = int(math.Abs(float64(corner2[i])-float64(corner1[i])) + 1)
	}

	region := NewTensor(newDimSizes, nil)

	for i := NewCoordIterator(corner1, corner2); i.HasNext(); {
		coords := i.Next()
		region.values[i.GetCurrentCount()-1] = t.AtCoords(coords)
	}

	return region
}

func (t *Tensor) Equals(other *Tensor) bool {
	if len(t.dimension) != len(other.dimension) {
		return false
	}
	for i := 0; i < len(t.dimension); i++ {
		if t.dimension[i] != other.dimension[i] {
			return false
		}
	}

	if len(t.values) != len(other.values) {
		return false
	}

	for i := 0; i < len(t.values); i++ {
		if fmt.Sprintf("%10.f", t.values[i]) != fmt.Sprintf("%10.f", other.values[i]) {
			return false
		}
	}

	return true
}

func (t *Tensor) InnerProduct(other *Tensor) float64 {
	if len(t.values) != len(other.values) {
		panic("Tensor.InnerProduct dimension mismatch")
	}
	result := 0.0
	for i := 0; i < len(t.values); i++ {
		result += t.values[i] * other.values[i]
	}
	return result
}

func (t *Tensor) MaxValueIndex() int {
	highestIndex := -1
	highestValue := math.MaxFloat64 * -1
	for i := 0; i < len(t.values); i++ {
		if highestValue < t.values[i] {
			highestValue = t.values[i]
			highestIndex = i
		}
	}
	return highestIndex
}
func (t *Tensor) MaxValue() float64 {
	highestValue := math.MaxFloat64 * -1
	for i := 0; i < len(t.values); i++ {
		if highestValue < t.values[i] {
			highestValue = t.values[i]
		}
	}
	return highestValue
}

func (t *Tensor) Flip() *Tensor {
	flippedTensor := NewTensor(t.dimension, nil)

	i := NewCoordIterator([]int{0, 0, 0}, func(l []int, r int) []int {
		ret := make([]int, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] + r
		}
		return ret
	}(t.dimension, -1))

	for i.HasNext() {
		currentCoords := i.Next()
		flippedCoords := make([]int, len(currentCoords))

		for j := 0; j < len(currentCoords); j++ {
			flippedCoords[j] = t.dimension[j] - 1 - currentCoords[j]
		}
		flippedTensor.Set(currentCoords, t.AtCoords(flippedCoords))
	}

	return flippedTensor
}

func (t *Tensor) Zeroes() *Tensor {
	return NewTensor(t.dimension, nil)
}
func (t *Tensor) AtCoords(coords []int) float64 {
	index := CoordsToHorner(coords, t.dimension)
	if index >= 0 && index < len(t.values) {
		return t.values[index]
	}
	return 0
}

func (t *Tensor) SetValue(idx int, val float64) { t.values[idx] = val }
func (t *Tensor) Set(coords []int, val float64) { t.values[CoordsToHorner(coords, t.dimension)] = val }
func (t *Tensor) Len() int                      { return len(t.values) }
func (t *Tensor) At(i int) float64              { return t.values[i] }
func (t *Tensor) Dimensions() []int             { return t.dimension }
func (t *Tensor) Values() []float64             { return t.values }
