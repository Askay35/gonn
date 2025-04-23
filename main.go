package main

import (
	"fmt"
	"time"

	// "time"

	nmath "github.com/Askay35/gonn/math"
	"github.com/Askay35/gonn/utils"
	"gonum.org/v1/gonum/mat"
)

const (
	PRINT_OUTPUT_EVERY = 250
)

func main() {
	var train_data utils.MNISTData
	train_data, err := utils.ReadMNISTData("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte")
	if err != nil {
		panic(err)
	}

	train_inputs := utils.BytesToFloats(train_data.Pixels)
	train_outputs := utils.LabelsToOuputs(train_data.Labels, OUTPUTS)

	var network Network

	//hidden layers
	for i := 0; i < HIDDEN_LAYERS; i++ {
		network.Layers = append(network.Layers, Layer{})

		network.Layers[i].Bias = mat.NewDense(1, HIDDEN_NEURONS, nil)
		if i == 0 {
			network.Layers[i].Weights = nmath.RandomDense(INPUTS, HIDDEN_NEURONS, -1, 1)
		} else {
			network.Layers[i].Weights = nmath.RandomDense(HIDDEN_NEURONS, HIDDEN_NEURONS, -1, 1)
		}
		network.Layers[i].Activation = "sigmoid"
		network.Layers[i].Output = mat.NewDense(1, HIDDEN_NEURONS, nil)

	}

	//output layer
	network.Layers = append(network.Layers, Layer{})

	network.Layers[len(network.Layers)-1].Bias = mat.NewDense(1, OUTPUTS, nil)
	network.Layers[len(network.Layers)-1].Weights = nmath.RandomDense(HIDDEN_NEURONS, OUTPUTS, -1, 1)
	network.Layers[len(network.Layers)-1].Activation = "softmax"
	network.Layers[len(network.Layers)-1].Output = mat.NewDense(1, OUTPUTS, nil)

	tries, right_answers, result := 0, 0, 0.0

	start := time.Now()
	for epoch := 0; epoch < EPOCHS; epoch++ {
		for i := 0; i < train_data.ImagesNumber; i++ {
			train_input := mat.NewDense(1, utils.IMAGE_SIZE_BYTES, train_inputs[i*utils.IMAGE_SIZE_BYTES:(i+1)*utils.IMAGE_SIZE_BYTES])

			output := network.Forward(train_input)

			train_output := mat.NewDense(1, OUTPUTS, train_outputs[i*OUTPUTS:(i+1)*OUTPUTS])

			answer, err := GetNetworkAnswer(output)

			if err != nil {
				fmt.Print(err, "\n")
				continue
			}

			tries++
			if train_data.Labels[i] == answer {
				right_answers++
			}

			if i%PRINT_OUTPUT_EVERY == 0 {
				e := nmath.CrossEntropy(train_output, output)
				result = float64(right_answers) / float64(tries) * 100.0

				utils.ClearConsole()
				utils.PrintMNIST(train_input.RawMatrix().Data)

				fmt.Printf("EPOCH: %v\n", epoch+1)
				fmt.Printf("IMAGE OUT - NETWORK OUT : %v - %v\n", train_data.Labels[i], answer)
				fmt.Printf("ERROR: %.4f%%\n", e)
				fmt.Printf("RIGHT ANSWERS: %.4f%% [ %d / %d ]\n", result, right_answers, tries)
			}

			network.Backpropogation(train_output)
		}

	}

	elapsed := time.Since(start) // get time elapsed
	fmt.Printf("Time elapsed: %s\n", elapsed)

}
