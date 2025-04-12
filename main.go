package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

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

func main() {
	input := mat.NewDense(2, 3, []float64{
		2.0, 1.0, 0.0,
		0.0, 3.0, 2.0,
	})
	probs := softmax(input)
	matPrint(input)
	matPrint(probs)
}

// Вспомогательная функция для печати матриц
func matPrint(X *mat.Dense) {
	f := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", f)
}
