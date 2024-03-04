package layers

import (
	"gonum.org/v1/gonum/mat"
)

type Dense struct {
	num_nodes     int
	activation_fn func(float64) float64
}

func (d *Dense) NodeCount() int {
	return d.num_nodes
}

func (d *Dense) Activation() func(float64) float64 {
	return d.activation_fn
}

func NewDense(num_nodes int, activation_fn func(float64) float64) *Dense {
	return &Dense{num_nodes, activation_fn}
}

func (d *Dense) IsInputLayer() bool {
	return false
}

func (d *Dense) IsOutputLayer() bool {
	return false
}

func (d *Dense) OutputFunction() func(*mat.Dense) *mat.Dense {
	return nil
}