package layers

import "gonum.org/v1/gonum/mat"

type Output struct {
	outputFunction func(*mat.Dense) *mat.Dense
}


func (o *Output) NodeCount() int {
	return 0
}

func (o *Output) Activation() func(float64) float64 {
	return nil
}

func (o *Output) IsInputLayer() bool {
	return false
}

func (o *Output) IsOutputLayer() bool {
	return true
}

func NewOutput(outputFunction func(*mat.Dense) *mat.Dense) *Output {
	return &Output{outputFunction}
}

func (o *Output) OutputFunction() func(*mat.Dense) *mat.Dense {
	return o.outputFunction
}


