package maths

type Tensor struct {
	dimension []int
	values    []float64
}

func NewTensor(dimension []int, values []float64) *Tensor {
	return &Tensor{dimension: dimension, values: values}
}
