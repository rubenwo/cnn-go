package layer

type Layer interface {
	ForwardPropagation()
	BackwardPropagation()

	OutputDims() []int
}
