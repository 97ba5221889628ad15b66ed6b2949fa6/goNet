package utils

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func GenerateRandomMatrix(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()*2 - 1
	}
	return mat.NewDense(rows, cols, data)
}
