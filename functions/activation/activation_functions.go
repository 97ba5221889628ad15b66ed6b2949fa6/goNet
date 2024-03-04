package activation

import (
	"math"
)

func ReLU(x float64) float64 {
	return max(0, x)
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func None(x float64) float64 {
	return x
}
