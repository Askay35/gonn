package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func getAnswer(o *mat.Dense) int {
	if o.At(0, 0) > o.At(0, 1) {
		return 0
	}
	return 1
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

// t - true values - правильные значение one-hot (1, 0, 0)...
// p - probs 	   - вероятности
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
		// Для численной стабильности (предотвращение переполнения)
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
