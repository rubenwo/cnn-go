package layer

import "github.com/rubenwo/cnn-go/pkg/cnn/maths"

type Layer interface {
	ForwardPropagation(input maths.Tensor) maths.Tensor
	BackwardPropagation(gradient maths.Tensor, lr float64) maths.Tensor

	OutputDims() []int
}
