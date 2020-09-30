package maths

import (
	"fmt"
	"math"
)

type CoordIteratorInterface interface {
	GetCurrentCoords() []int
	GetCurrentCount() int
	HasNext() bool
	Next() []int
}
type CoordIterator struct {
	startCoords, finalCoords, currentCoords []int
	currentCount                            int
	finalIndex                              int
}

func NewCoordIterator(corner1, corner2 []int) *CoordIterator {
	it := &CoordIterator{
		startCoords:   make([]int, len(corner1)),
		currentCoords: make([]int, len(corner1)),
		finalCoords:   make([]int, len(corner2)),
		finalIndex:    1,
	}

	for i := 0; i < len(it.startCoords); i++ {
		it.startCoords[i] = int(math.Min(float64(corner1[i]), float64(corner2[i])))
		it.currentCoords[i] = it.startCoords[i]
		it.finalCoords[i] = int(math.Max(float64(corner1[i]), float64(corner2[i]))) - 1

		it.finalIndex *= (it.finalCoords[i] - it.startCoords[i]) + 2
	}
	it.currentCoords[0]--
	return it
}

func (it *CoordIterator) GetCurrentCoords() []int { return it.currentCoords }
func (it *CoordIterator) GetCurrentCount() int    { return it.currentCount }
func (it *CoordIterator) HasNext() bool           { return it.currentCount < it.finalIndex }
func (it *CoordIterator) Next() []int {
	for i := 0; i < len(it.currentCoords); i++ {
		if it.currentCoords[i] > it.finalCoords[i] {
			it.currentCount += it.currentCoords[i] - it.finalCoords[i] - 1
			it.currentCoords[i] = it.startCoords[i]
		} else {
			it.currentCoords[i] = it.currentCoords[i] + 1
			it.currentCount++
			return it.currentCoords
		}
	}
	panic("Tried to access coordinate beyond boundary")
}

type StridingCoordIterator struct {
	strides, corner1, currentCoords []int
	coordIter                       *CoordIterator
}

func NewStridingCoordIterator(corner1, corner2, strides []int) *StridingCoordIterator {
	it := &StridingCoordIterator{corner1: corner1, strides: strides}
	cornerDiff := func(l, r []int) []int {
		ret := make([]int, len(l))
		for i := 0; i < len(l); i++ {
			// left minus right
			ret[i] = l[i] - r[i]
		}
		return ret
	}(corner2, corner1)

	adjustedDifference := func(l, r []int) []int {
		ret := make([]int, len(l))
		for i := 0; i < len(l); i++ {
			// left divided right
			ret[i] = l[i] / r[i]
		}
		return ret
	}(cornerDiff, strides)

	it.coordIter = NewCoordIterator(make([]int, len(corner1)), adjustedDifference)
	return it
}
func (it *StridingCoordIterator) GetCurrentCoords() []int { return it.currentCoords }
func (it *StridingCoordIterator) GetCurrentCount() int    { return it.coordIter.GetCurrentCount() }
func (it *StridingCoordIterator) HasNext() bool           { return it.coordIter.HasNext() }
func (it *StridingCoordIterator) Next() []int {
	mult := func(l, r []int) []int {
		ret := make([]int, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] * r[i]
		}
		return ret
	}(it.coordIter.Next(), it.strides)

	it.currentCoords = func(l, r []int) []int {
		ret := make([]int, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] + r[i]
		}
		return ret
	}(mult, it.corner1)
	return it.currentCoords
}

type RegionsIterator struct {
	regionSizes  []int
	bottomCorner []int
	topCorner    []int

	CoordIterator CoordIteratorInterface

	tensor *Tensor
}

func (it *RegionsIterator) setup(regionSizes, padding []int) {
	regionSizeCopy := regionSizes

	if len(regionSizeCopy) < len(it.tensor.dimension) {
		regionSizeCopy = make([]int, len(it.tensor.dimension))
		for i := 0; i < len(regionSizes); i++ {
			regionSizeCopy[i] = regionSizes[i]
		}
		for i := len(regionSizes); i < len(it.tensor.dimension); i++ {
			regionSizeCopy[i] = 1
		}
	}

	it.regionSizes = func(r []int, f int) []int {
		ret := make([]int, len(r))
		for i := 0; i < len(ret); i++ {
			ret[i] = r[i] + f
		}
		return ret
	}(regionSizeCopy, -1)

	it.bottomCorner = make([]int, len(it.tensor.dimension))

	it.bottomCorner = func(r, p []int) []int {
		ret := make([]int, len(r))
		for i := 0; i < len(ret); i++ {
			if len(p) > i {
				ret[i] = r[i] - p[i]
			} else {
				ret[i] = r[i]
			}
		}
		return ret
	}(it.bottomCorner, padding)

	it.topCorner = func(r, p []int) []int {
		ret := make([]int, len(r))
		for i := 0; i < len(ret); i++ {
			if len(p) > i {
				ret[i] = r[i] - p[i]
			} else {
				fmt.Println("Should prolly not happen")
				ret[i] = r[i]
			}
		}
		return ret
	}(it.tensor.dimension, regionSizeCopy)

	it.topCorner = func(r, p []int) []int {
		ret := make([]int, len(r))
		for i := 0; i < len(ret); i++ {
			if len(p) > i {
				ret[i] = r[i] + p[i]
			} else {
				ret[i] = r[i]
			}
		}
		return ret
	}(it.topCorner, padding)
}

func NewRegionsIterator(t *Tensor, regionSizes, padding []int) *RegionsIterator {
	it := &RegionsIterator{tensor: t}
	it.setup(regionSizes, padding)
	it.CoordIterator = NewCoordIterator(it.bottomCorner, it.topCorner)
	return it
}

func NewRegionsIteratorWithStrides(t *Tensor, regionSizes, padding, strides []int) *RegionsIterator {
	it := &RegionsIterator{tensor: t}
	it.setup(regionSizes, padding)
	it.CoordIterator = NewStridingCoordIterator(it.bottomCorner, it.topCorner, strides)
	return it
}

func (it *RegionsIterator) HasNext() bool {
	return it.CoordIterator.HasNext()
}

func (it *RegionsIterator) Next() *Tensor {
	regionBottomCorner := it.CoordIterator.Next()
	regionTopCorner := func(r, p []int) []int {
		ret := make([]int, len(r))
		for i := 0; i < len(ret); i++ {
			if len(p) > i {
				ret[i] = r[i] + p[i]
			} else {
				ret[i] = r[i]
			}
		}
		return ret
	}(regionBottomCorner, it.regionSizes)

	return it.tensor.Region(regionBottomCorner, regionTopCorner)
}

type RegionsIteratorIterator struct {
	iter *RegionsIterator
}

func NewRegionsIteratorIterator(tensor *Tensor, regionSizes, padding []int) *RegionsIteratorIterator {
	it := &RegionsIteratorIterator{iter: NewRegionsIterator(tensor, regionSizes, padding)}
	return it
}

func (it *RegionsIteratorIterator) HasNext() bool { return it.iter.HasNext() }
func (it *RegionsIteratorIterator) Next() *ValuesIterator {
	regionBottomCorner := it.iter.CoordIterator.Next()
	regionTopCorner := func(l, r []int) []int {
		ret := make([]int, len(l))
		for i := 0; i < len(ret); i++ {
			ret[i] = l[i] + r[i]
		}
		return ret
	}(regionBottomCorner, it.iter.regionSizes)

	return NewValuesIterator(it.iter.tensor, NewCoordIterator(regionBottomCorner, regionTopCorner))
}

type ValuesIterator struct {
	iter   *CoordIterator
	tensor *Tensor
}

func NewValuesIterator(tensor *Tensor, iter *CoordIterator) *ValuesIterator {
	return &ValuesIterator{tensor: tensor, iter: iter}
}

func (it *ValuesIterator) HasNext() bool { return it.iter.HasNext() }
func (it *ValuesIterator) Next() float64 { return it.tensor.AtCoords(it.iter.Next()) }

func (it *ValuesIterator) InnerProduct(t *Tensor) float64 {
	result := 0.0
	for i := 0; i < len(t.values); i++ {
		result += it.Next() * t.values[i]
	}
	return result
}
