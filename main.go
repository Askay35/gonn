package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

const (
	INPUTS             = 784
	HIDDEN_NEURONS     = 200
	HIDDEN_LAYERS      = 1
	OUTPUTS            = 10
	LEARNING_RATE      = 0.01
	EPOCHS             = 500
	PRINT_OUTPUT_EVERY = 500
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

	var train_data MNISTData
	train_data, err := readMNISTData("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Loaded %d images. Size %dx%d\n", train_data.images_number, IMAGE_WIDTH, IMAGE_HEIGHT)
	fmt.Printf("Loaded %d labels\n", len(train_data.labels))

	resetData()

	train_inputs := bytesToFloats(train_data.pixels)
	train_outputs := labelsToOuputs(train_data.labels, OUTPUTS)

	for epoch := 0; epoch < EPOCHS; epoch++ {

		for i := 0; i < train_data.images_number; i++ {
			train_input := mat.NewDense(1, IMAGE_SIZE_BYTES, train_inputs[i*IMAGE_SIZE_BYTES:(i+1)*IMAGE_SIZE_BYTES])

			if input_hidden_bias.RawMatrix().Rows == INPUTS {
				hiddens_pre.Add(input_hidden_weights.T(), &input_hidden_bias)
			} else {
				hiddens_pre = *addVectorToMatrix(mat.DenseCopyOf(input_hidden_weights.T()), &input_hidden_bias)
			}

			hiddens.Mul(train_input, &hiddens_pre)
			hiddens = *sigmoid(&hiddens)

			if hidden_output_bias.RawMatrix().Rows == HIDDEN_NEURONS {
				outputs_pre.Add(hidden_output_weights.T(), &hidden_output_bias)
			} else {
				outputs_pre = *addVectorToMatrix(mat.DenseCopyOf(hidden_output_weights.T()), &hidden_output_bias)
			}

			outputs.Mul(&hiddens, &outputs_pre)
			outputs = *softmax(&outputs)

			train_output := mat.NewDense(1, OUTPUTS, train_outputs[i*OUTPUTS:(i+1)*OUTPUTS])

			answer, err := getNetworkAnswer(&outputs)
			if err != nil {
				fmt.Print(err, "\n")
				continue
			}

			if i%PRINT_OUTPUT_EVERY == 0 {
				printMNIST(train_input.RawMatrix().Data)
				fmt.Printf("IMAGE OUT - NETWORK OUT : %v - %v\n", train_data.labels[i], answer)
				e := crossEntropy(train_output, &outputs)
				fmt.Printf("ERROR: %v\n", e)
				fmt.Printf("EPOCH: %v\n", epoch+1)
			}

			delta_output.Sub(&outputs, train_output)
			hiddens_adj = *multiplyMatrix(mat.DenseCopyOf(hiddens.T()), &delta_output)
			hiddens_adj.Scale(-LEARNING_RATE, &hiddens_adj)

			hidden_output_weights.Add(&hidden_output_weights, hiddens_adj.T())
			if hidden_output_bias.RawMatrix().Rows == HIDDEN_NEURONS {
				hidden_output_bias.Add(&hiddens_adj, &hidden_output_bias)
			} else {
				hidden_output_bias = *addVectorToMatrix(&hiddens_adj, &hidden_output_bias)
			}

			hiddens_derivative = *sigmoidDerivative(&hiddens)
			var temp mat.Dense

			temp.Mul(&delta_output, &hidden_output_weights)
			delta_hidden.MulElem(&temp, &hiddens_derivative)
			inputs_adj.Mul(train_input.T(), &delta_hidden)
			inputs_adj.Scale(-LEARNING_RATE, &inputs_adj)

			input_hidden_weights.Add(&input_hidden_weights, inputs_adj.T())

			if input_hidden_bias.RawMatrix().Rows == INPUTS {
				input_hidden_bias.Add(&inputs_adj, &input_hidden_bias)
			} else {
				input_hidden_bias = *addVectorToMatrix(&inputs_adj, &input_hidden_bias)
			}

			hiddens_adj.Reset()
			inputs_adj.Reset()
		}

	}

}
