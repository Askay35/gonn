package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

const (
	INPUTS         = 3
	HIDDEN_NEURONS = 3
	HIDDEN_LAYERS  = 1
	OUTPUTS        = 2
	LEARNING_RATE  = 0.5
	EPOCHS         = 500
)

var input_hidden_weights, hidden_output_weights, inputs_adj, hiddens_adj, hiddens_derivative, hiddens_pre, outputs_pre, hiddens, outputs, delta_output, delta_hidden mat.Dense
var input_hidden_bias, hidden_output_bias mat.Dense

func resetData() {
	input_hidden_bias = *mat.NewDense(1, HIDDEN_NEURONS, nil)
	hidden_output_bias = *mat.NewDense(1, OUTPUTS, nil)

	input_hidden_weights = *randomDense(HIDDEN_NEURONS, INPUTS, -1, 1)
	hidden_output_weights = *randomDense(OUTPUTS, HIDDEN_NEURONS, -1, 1)
}

func main() {

	var train_inputs = [][]float64{
		{0, 0, 1},
		{1, 1, 1},
		{1, 0, 1},
		{0, 1, 1},
	}

	var train_outputs = [][]float64{
		{1, 0},
		{0, 1},
		{0, 1},
		{1, 0},
	}

	resetData()

	for epoch := 0; epoch < EPOCHS; epoch++ {
		// fmt.Printf("epoch: %v\n", epoch+1)

		for input_index, input := range train_inputs {

			train_input := mat.NewDense(1, len(input), input)
			if input_hidden_bias.RawMatrix().Rows == 3 {
				hiddens_pre.Add(&input_hidden_weights, &input_hidden_bias)
			} else {
				hiddens_pre = *addVectorToMatrix(&input_hidden_weights, &input_hidden_bias)
			}

			hiddens.Mul(train_input, &hiddens_pre)
			hiddens = *sigmoid(&hiddens)

			if input_hidden_bias.RawMatrix().Rows == 3 {
				outputs_pre.Add(hidden_output_weights.T(), &hidden_output_bias)
			} else {
				outputs_pre = *addVectorToMatrix(mat.DenseCopyOf(hidden_output_weights.T()), &hidden_output_bias)
			}

			outputs.Mul(&hiddens, &outputs_pre)
			outputs = *softmax(&outputs)

			train_output := mat.NewDense(1, OUTPUTS, train_outputs[input_index])

			if train_output.At(0, 0) == 0 {
				fmt.Print("right output: 1\n")
			} else {
				fmt.Print("right output: 0\n")
			}

			fmt.Printf("output: %v\n", getAnswer(&outputs))

			// cost / error
			// e := crossEntropy(train_output, &outputs)
			// fmt.Printf("error: %v\n\n", e)

			// back propogation

			delta_output.Sub(&outputs, train_output)
			hiddens_adj = *multiplyMatrix(mat.DenseCopyOf(hiddens.T()), &delta_output)
			hiddens_adj.Scale(-LEARNING_RATE, &hiddens_adj)

			hidden_output_weights.Add(&hidden_output_weights, hiddens_adj.T())
			if hidden_output_bias.RawMatrix().Rows == 3 {
				hidden_output_bias.Add(&hiddens_adj, &hidden_output_bias)
			} else {
				hidden_output_bias = *addVectorToMatrix(&hiddens_adj, &hidden_output_bias)
			}

			hiddens_derivative = *sigmoidDerivative(&hiddens)
			var temp mat.Dense

			temp.Mul(&delta_output, &hidden_output_weights) // (1×2) × (2×3) = (1×3)
			delta_hidden.MulElem(&temp, &hiddens_derivative)
			inputs_adj.Mul(train_input.T(), &delta_hidden)
			inputs_adj.Scale(-LEARNING_RATE, &inputs_adj)

			input_hidden_weights.Add(&input_hidden_weights, &inputs_adj)

			if input_hidden_bias.RawMatrix().Rows == 3 {
				input_hidden_bias.Add(&inputs_adj, &input_hidden_bias)
			} else {
				input_hidden_bias = *addVectorToMatrix(&inputs_adj, &input_hidden_bias)
			}

			hiddens_adj.Reset()
			inputs_adj.Reset()
		}

	}

}
