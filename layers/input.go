package layers

import (
	"nn/functions/activation"

	"gonum.org/v1/gonum/mat"
)

type Input struct {
	num_nodes     int
}

func (i *Input) NodeCount() int {
	return i.num_nodes
}

func (i *Input) IsInputLayer() bool {
	return true
}

func (i *Input) IsOutputLayer() bool {
	return false
}

func NewInput(num_nodes int) *Input {
	return &Input{num_nodes}
}

func (i *Input) Activation() func(float64) float64 {
	return activation.None
}

func (i *Input) OutputFunction() func(*mat.Dense) *mat.Dense {
	return nil
}

