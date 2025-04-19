package main

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/mat"
)

const (
	PRINT_OUTPUT_EVERY = 300
)

func main() {
	var train_data MNISTData
	train_data, err := readMNISTData("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte")
	if err != nil {
		panic(err)
	}

	train_inputs := bytesToFloats(train_data.pixels)
	train_outputs := labelsToOuputs(train_data.labels, OUTPUTS)

	var network Network

	for i := 0; i < HIDDEN_LAYERS; i++ {
		network.Layers = append(network.Layers, Layer{})
	}
	//output layer
	network.Layers = append(network.Layers, Layer{})

	network.Layers[0].Bias = mat.NewDense(1, HIDDEN_NEURONS, nil)
	network.Layers[0].Weights = randomDense(INPUTS, HIDDEN_NEURONS, -1, 1)
	network.Layers[0].Activation = "sigmoid"
	network.Layers[0].Output = mat.NewDense(1, HIDDEN_NEURONS, nil)

	network.Layers[1].Bias = mat.NewDense(1, HIDDEN_NEURONS, nil)
	network.Layers[1].Weights = randomDense(HIDDEN_NEURONS, HIDDEN_NEURONS, -1, 1)
	network.Layers[1].Activation = "sigmoid"
	network.Layers[1].Output = mat.NewDense(1, HIDDEN_NEURONS, nil)

	network.Layers[2].Bias = mat.NewDense(1, OUTPUTS, nil)
	network.Layers[2].Weights = randomDense(HIDDEN_NEURONS, OUTPUTS, -1, 1)
	network.Layers[2].Activation = "softmax"
	network.Layers[2].Output = mat.NewDense(1, OUTPUTS, nil)

	tries, right_answers, result := 0, 0, 0.0

	start := time.Now()

	for epoch := 0; epoch < EPOCHS; epoch++ {
		for i := 0; i < train_data.images_number; i++ {
			train_input := mat.NewDense(1, IMAGE_SIZE_BYTES, train_inputs[i*IMAGE_SIZE_BYTES:(i+1)*IMAGE_SIZE_BYTES])

			output := network.forward(train_input)

			train_output := mat.NewDense(1, OUTPUTS, train_outputs[i*OUTPUTS:(i+1)*OUTPUTS])

			answer, err := getNetworkAnswer(output)

			if err != nil {
				fmt.Print(err, "\n")
				continue
			}

			tries++
			if train_data.labels[i] == answer {
				right_answers++
			}

			if i%PRINT_OUTPUT_EVERY == 0 {
				clearConsole()
				printMNIST(train_input.RawMatrix().Data)
				fmt.Printf("IMAGE OUT - NETWORK OUT : %v - %v\n", train_data.labels[i], answer)
				e := crossEntropy(train_output, output)
				fmt.Printf("ERROR: %v\n", e)
				fmt.Printf("EPOCH: %v\n", epoch+1)
				result = float64(right_answers) / float64(tries) * 100.0
				fmt.Printf("RIGHT ANSWERS: %f%%\n", result)
			}

			network.backpropogation(train_output)
		}

	}

	elapsed := time.Since(start) // Получаем продолжительность
	fmt.Printf("Время выполнения: %s\n", elapsed)

}
