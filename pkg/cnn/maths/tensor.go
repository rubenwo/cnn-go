package maths

type Tensor struct {
	dimension []int
	values    []float64
}

func NewTensor(dimension []int, values []float64) *Tensor {
	return &Tensor{dimension: dimension, values: values}
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

func (t *Tensor) Len() int          { return len(t.values) }
func (t *Tensor) At(i int) float64  { return t.values[i] }
func (t *Tensor) Dimensions() []int { return t.dimension }
func (t *Tensor) Values() []float64 { return t.values }
