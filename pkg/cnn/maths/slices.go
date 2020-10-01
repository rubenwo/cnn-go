package maths

import "math"

func AddIntSlices(l, r []int) []int {
	if len(l) != len(r) {
		panic("AddIntSlices len(l) != len(r)")
	}
	ret := make([]int, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] + r[i]
	}
	return ret
}
func AddIntToAll(l []int, r int) []int {
	ret := make([]int, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] + r
	}
	return ret
}
func SubtractIntSlices(l, r []int) []int {
	if len(l) != len(r) {
		panic("SubtractIntSlices len(l) != len(r)")
	}
	ret := make([]int, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] - r[i]
	}
	return ret
}
func DivideIntSlices(l, r []int) []int {
	if len(l) != len(r) {
		panic("DivideIntSlices len(l) != len(r)")
	}
	ret := make([]int, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] / r[i]
	}
	return ret
}
func MulIntSlices(l, r []int) []int {
	if len(l) != len(r) {
		panic("MulIntSlices len(l) != len(r)")
	}
	ret := make([]int, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] * r[i]
	}
	return ret
}

func ProductIntSlice(arr []int) int {
	product := 1
	for i := 0; i < len(arr); i++ {
		product *= arr[i]
	}
	return product
}

func AddFloat64Slices(l, r []float64) []float64 {
	if len(l) != len(r) {
		panic("AddFloat64Slices len(l) != len(r)")
	}
	ret := make([]float64, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] + r[i]
	}
	return ret
}
func AddFloat64ToSlice(l []float64, r float64) []float64 {
	ret := make([]float64, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] + r
	}
	return ret
}
func MulFloat64Slices(l, r []float64) []float64 {
	if len(l) != len(r) {
		panic("MulFloat64Slices len(l) != len(r)")
	}
	ret := make([]float64, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] * r[i]
	}
	return ret
}
func MulFloat64ToSlice(l []float64, r float64) []float64 {
	ret := make([]float64, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] * r
	}
	return ret
}

func DivideFloat64Slices(l, r []float64) []float64 {
	if len(l) != len(r) {
		panic("DivideFloat64Slices len(l) != len(r)")
	}
	ret := make([]float64, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] / r[i]
	}
	return ret
}
func DivideFloat64SliceByFloat64(l []float64, r float64) []float64 {
	ret := make([]float64, len(l))
	for i := 0; i < len(ret); i++ {
		ret[i] = l[i] / r
	}
	return ret
}

func ProductFloat64Slice(arr []float64) float64 {
	product := 1.0
	for i := 0; i < len(arr); i++ {
		product *= arr[i]
	}
	return product
}

func SumFloat64Slice(arr []float64) float64 {
	sum := 0.0
	for i := 0; i < len(arr); i++ {
		sum += arr[i]
	}
	return sum
}

func FindMaxIndexFloat64Slice(arr []float64) int {
	highest := math.MaxFloat64 * -1
	highestIndex := -1

	for idx, val := range arr {
		if val > highest {
			highest = val
			highestIndex = idx
		}
	}

	return highestIndex
}
