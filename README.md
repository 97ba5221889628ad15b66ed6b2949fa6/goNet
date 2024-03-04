# Go Neural Network
A simple proof of concept neural network library written in Go.


## Installation
`git clone REPO`


## Examples
```go
import (
	"fmt"
	"nn/functions/activation"
	"nn/functions/output"
	"nn/layers"
	"nn/network"
	"nn/utils"

	"gonum.org/v1/gonum/mat"
)

func main() {
	nn := network.NewNeuralNetwork()
	nn.AddLayerbyLayer(layers.NewInput(10)).AddLayerbyLayer(layers.NewDense(64, activation.ReLU)).AddLayerbyLayer(layers.NewDense(32, activation.ReLU)).AddLayerbyLayer(layers.NewDense(10, activation.ReLU)).AddLayerbyLayer(layers.NewOutput(output.Max))
	nn.Compile()

	input := utils.GenerateRandomMatrix(10, 1)

	result := nn.Forward(input)

	fmt.Println(mat.Formatted(result, mat.Prefix("")))
}
```


