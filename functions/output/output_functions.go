package output

import (
	"gonum.org/v1/gonum/mat"
)

func Softmax(input *mat.Dense) *mat.Dense {
	panic("Softmax not implemented")
}

func None(input *mat.Dense) *mat.Dense {
	return input
}

func Max(input *mat.Dense) *mat.Dense {
	r, _ := input.Dims()

	maxVal := input.At(0, 0)

	for i := 0; i < r; i++ {
		val := input.At(i, 0)
		if val > maxVal {
			maxVal = val
		}
	}

	output := mat.NewDense(1, 1, []float64{maxVal})

	return output
}
