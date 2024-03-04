package network

import (
	"errors"

	"nn/forward"
	"nn/utils"

	"gonum.org/v1/gonum/mat"
)

type Layer interface {
	NodeCount() int

	Activation() func(float64) float64

	IsInputLayer() bool

	IsOutputLayer() bool

	OutputFunction() func(*mat.Dense) *mat.Dense
}

type NeuralNetwork struct {
	weights        []mat.Dense
	biases         []mat.Dense
	activation     []func(float64) float64
	inputDim       int
	outputFunction func(*mat.Dense) *mat.Dense
	layers         []Layer
}

func NewNeuralNetwork() *NeuralNetwork {
	return &NeuralNetwork{}
}

func (nn *NeuralNetwork) SetInputLayer(layerSize int) error {
	if nn.inputDim != 0 {
		return errors.New("input layer already exists")
	}

	nn.inputDim = layerSize
	return nil
}

func (nn *NeuralNetwork) AddLayerbyLayer(layer Layer) *NeuralNetwork {
	nn.layers = append(nn.layers, layer)
	return nn
}

func (nn *NeuralNetwork) Compile() {
	if len(nn.layers) == 0 {
		panic("no layers added")
	}
	if len(nn.layers) == 1 {
		panic("only one layer added")
	}
	if nn.layers[0].IsInputLayer() {
		nn.inputDim = nn.layers[0].NodeCount()
	} else {
		panic("first layer must be input layer")
	}

	if nn.layers[len(nn.layers)-1].IsOutputLayer() {
		nn.SetOutputFunction(nn.layers[len(nn.layers)-1].OutputFunction())
		if nn.outputFunction == nil {
			panic("output function not set")
		}
	} else {
		panic("last layer must be output layer")
	}

	if len(nn.weights) != 0 || len(nn.biases) != 0 {
		panic("weights and biases must be empty")
	}

	for i := 1; i < len(nn.layers)-1; i++ {
		nn.weights = append(nn.weights, *utils.GenerateRandomMatrix(nn.layers[i].NodeCount(), nn.layers[i-1].NodeCount()))
		nn.biases = append(nn.biases, *utils.GenerateRandomMatrix(nn.layers[i].NodeCount(), 1))
		nn.activation = append(nn.activation, nn.layers[i].Activation())
	}

}

func (nn *NeuralNetwork) AddLayer(layerSize int, activation func(float64) float64) {
	if len(nn.weights) == 0 {
		nn.weights = append(nn.weights, *utils.GenerateRandomMatrix(layerSize, nn.inputDim))
		nn.biases = append(nn.biases, *utils.GenerateRandomMatrix(layerSize, 1))
		nn.activation = append(nn.activation, activation)
		return
	}
	nn.weights = append(nn.weights, *utils.GenerateRandomMatrix(layerSize, nn.weights[len(nn.weights)-1].RawMatrix().Rows))
	nn.biases = append(nn.biases, *utils.GenerateRandomMatrix(layerSize, 1))
	nn.activation = append(nn.activation, activation)
}

func (nn *NeuralNetwork) Forward(inputs *mat.Dense) *mat.Dense {
	if inputs.RawMatrix().Cols != 1 {
		panic("Input matrix must have 1 column")
	}
	if inputs.RawMatrix().Rows != nn.inputDim {
		panic("Input matrix must have the same number of rows as the input layer")
	}
	if nn.outputFunction == nil {
		panic("Output function not set")
	}

	result := forward.FeedForwardAll(inputs, nn.weights, nn.biases, nn.activation)

	result = nn.outputFunction(result)

	return result
}

func (nn *NeuralNetwork) SetOutputFunction(f func(*mat.Dense) *mat.Dense) {
	nn.outputFunction = f
}
