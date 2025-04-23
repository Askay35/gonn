package main

import (
	"fmt"

	nmath "github.com/Askay35/gonn/math"
	"gonum.org/v1/gonum/mat"
)

const (
	INPUTS         = 784 //28x28 mnist image size
	HIDDEN_NEURONS = 50
	HIDDEN_LAYERS  = 1
	OUTPUTS        = 10 // 10 digits - 0-9
	LEARNING_RATE  = 0.01
	EPOCHS         = 1
)

type Layer struct {
	Bias, Weights, Input, Output, Z *mat.Dense
	Activation                      string //activation type: sigmoid, softmax, relu ...
}

func (l *Layer) forward(input *mat.Dense) *mat.Dense {

	l.Input = input

	if input.RawMatrix().Cols == l.Weights.RawMatrix().Rows {
		l.Z = nmath.MultiplyMatrix(input, l.Weights)
	} else {
		l.Z = nmath.MultiplyMatrix(input, mat.DenseCopyOf(l.Weights.T()))
	}

	if l.Z.RawMatrix().Rows == l.Bias.RawMatrix().Rows && l.Z.RawMatrix().Cols == l.Bias.RawMatrix().Cols {
		l.Z.Add(l.Z, l.Bias)
	} else {
		l.Z = nmath.AddVectorToMatrix(mat.DenseCopyOf(l.Z.T()), l.Bias)
	}

	switch l.Activation {
	case "sigmoid":
		l.Output = nmath.Sigmoid(l.Z)
	case "softmax":
		l.Output = nmath.Softmax(l.Z)
	case "relu":
		l.Output = nmath.Relu(l.Z)
	default:
		l.Output = mat.DenseCopyOf(l.Z)
	}

	return l.Output
}

type Network struct {
	Layers []Layer
}

func (n *Network) Forward(input *mat.Dense) *mat.Dense {
	var output *mat.Dense
	for index := range n.Layers {
		if index == 0 {
			output = n.Layers[index].forward(input)
		} else {
			output = n.Layers[index].forward(output)
		}
	}
	return output
}
func (n *Network) Backpropogation(true_output *mat.Dense) {
	var delta, adj, temp mat.Dense

	for i := len(n.Layers) - 1; i >= 0; i-- {
		layer := n.Layers[i]
		switch layer.Activation {
		case "softmax":

			output_layer := n.Layers[len(n.Layers)-1]
			delta.Sub(output_layer.Output, true_output)
			adj = *nmath.MultiplyMatrix(mat.DenseCopyOf(layer.Input.T()), &delta)
			adj.Scale(-LEARNING_RATE, &adj)
			temp.Scale(-LEARNING_RATE, &delta)
			layer.Weights.Add(layer.Weights, &adj)
			if layer.Bias.RawMatrix().Rows == temp.RawMatrix().Rows {
				layer.Bias.Add(&temp, layer.Bias)
			} else {
				layer.Bias = nmath.AddVectorToMatrix(&temp, layer.Bias)
			}

		case "sigmoid":

			derivative := nmath.SigmoidDerivative(layer.Output)
			if i == len(n.Layers)-1 {
				delta.Sub(layer.Output, true_output)
			} else {
				delta = *nmath.MultiplyMatrix(&delta, mat.DenseCopyOf(n.Layers[i+1].Weights.T())) // delta [batch × next_size] × weightsT [next_size × current_size] → [batch × current_size]
			}
			delta.MulElem(&delta, derivative)
			adj = *nmath.MultiplyMatrix(mat.DenseCopyOf(layer.Input.T()), &delta)
			adj.Scale(-LEARNING_RATE, &adj)
			layer.Weights.Add(layer.Weights, &adj)

			if layer.Bias.RawMatrix().Rows == adj.RawMatrix().Rows {
				layer.Bias.Add(&adj, layer.Bias)
			} else {
				layer.Bias = nmath.AddVectorToMatrix(&adj, layer.Bias)
			}

			// case "relu":
			// 	derivative := nmath.SigmoidDerivative(layer.Output)
			// 	if i == len(n.Layers)-1 {
			// 		delta.Sub(layer.Output, true_output)
			// 		delta.MulElem(&delta, derivative)
			// 	} else {
			// 		delta = *nmath.MultiplyMatrix(&delta, mat.DenseCopyOf(n.Layers[i+1].Weights.T())) // delta [batch × next_size] × weightsT [next_size × current_size] → [batch × current_size]
			// 		delta.MulElem(&delta, derivative)
			// 	}
			// 	adj = *nmath.MultiplyMatrix(mat.DenseCopyOf(layer.Input.T()), &delta)
			// 	adj.Scale(-LEARNING_RATE, &adj)
			// 	layer.Weights.Add(layer.Weights, &adj)

			// 	if layer.Bias.RawMatrix().Rows == adj.RawMatrix().Rows {
			// 		layer.Bias.Add(&adj, layer.Bias)
			// 	} else {
			// 		layer.Bias = nmath.AddVectorToMatrix(&adj, layer.Bias)
			// 	}
		}

	}
}

func GetNetworkAnswer(output *mat.Dense) (byte, error) {
	rows, cols := output.Dims()
	if rows != 1 {
		return 0, fmt.Errorf("output matrix must have exactly 1 row to get answer")
	}

	matData := output.RawMatrix().Data

	maxValue := matData[0]
	maxIndex := 0

	for j := 1; j < cols; j++ {
		if matData[j] > maxValue {
			maxValue = matData[j]
			maxIndex = j
		}
	}

	return byte(maxIndex), nil
}
