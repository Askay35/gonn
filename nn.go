package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

func printMNIST(data []float64) {
	if len(data) != 28*28 {
		panic("Array length should be 784 bytes (28x28)")
	}

	gradient := []string{" ", "░", "▒", "▓", "█"}

	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {

			val := 255 - data[y*28+x]
			if val < 0 {
				val = 0
			}
			if val > 255 {
				val = 255
			}

			index := int(math.Round(val / 255 * float64(len(gradient)-1)))
			fmt.Print(gradient[index])
		}
		fmt.Print("\n")
	}
}

func getNetworkAnswer(output *mat.Dense) (int, error) {
	rows, cols := output.Dims()
	if rows != 1 {
		return 0, fmt.Errorf("Output matrix must have exactly 1 row to get answer")
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

	return maxIndex, nil
}

func softmax(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	rows_sums := []float64{}
	row_sum := 0.0

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			row_sum += math.Pow(math.E, m.At(i, j))
		}
		rows_sums = append(rows_sums, row_sum)
		row_sum = 0.0
	}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, math.Pow(math.E, m.At(i, j))/rows_sums[i])
		}
	}
	return result
}

func softmaxDerivative(t *mat.Dense, m *mat.Dense) *mat.Dense {
	var result mat.Dense
	result.Sub(m, t)
	return &result
}

func crossEntropy(t *mat.Dense, p *mat.Dense) float64 {
	r, c := p.Dims()
	result := 0.0

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result += t.At(i, j) * math.Log(p.At(i, j))
		}
	}

	return -result
}

func sigmoid(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return 1 / (1 + math.Exp(-v))
		}
		return math.Exp(v) / (1 + math.Exp(v))
	}, m)
	return result
}

func sigmoidDerivative(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(func(i, j int, v float64) float64 {
		s := 1 / (1 + math.Exp(-v))
		return s * (1 - s)
	}, m)
	return result
}
