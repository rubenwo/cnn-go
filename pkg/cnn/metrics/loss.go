package metrics

type LossFunction interface {
	CalculateLoss()
	CalculateLossDerivative()
}

type CrossEntropyLoss struct{}

func (c *CrossEntropyLoss) CalculateLoss()           {}
func (c *CrossEntropyLoss) CalculateLossDerivative() {}
