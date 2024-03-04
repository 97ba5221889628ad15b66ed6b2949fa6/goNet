package forward

import (
	"gonum.org/v1/gonum/mat"
)

func feedforwardOne(input, weights, biases *mat.Dense, activation func(float64) float64) *mat.Dense {
	var result mat.Dense
	result.Mul(weights, input)
	result.Add(&result, biases)

	result.Apply(func(i, j int, v float64) float64 {
		return activation(v)
	}, &result)

	return &result
}

func FeedForwardAll(inputs *mat.Dense, weights []mat.Dense, biases []mat.Dense, activation []func(float64) float64) *mat.Dense {
	if len(weights) != len(biases) {
		panic("weights and biases must be the same length")
	}
	if inputs.RawMatrix().Rows != weights[0].RawMatrix().Cols {
		panic("inputs and weights must be compatible")
	}

	var result mat.Dense
	result = *inputs

	for i := range weights {
		result = *feedforwardOne(&result, &weights[i], &biases[i], activation[i])
	}

	return &result
}
