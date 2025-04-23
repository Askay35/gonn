package math

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func Softmax(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		max := -math.MaxFloat64
		for j := 0; j < c; j++ {
			if v := m.At(i, j); v > max {
				max = v
			}
		}

		sum := 0.0
		row := make([]float64, c)
		for j := 0; j < c; j++ {
			row[j] = math.Exp(m.At(i, j) - max)
			sum += row[j]
		}

		for j := 0; j < c; j++ {
			result.Set(i, j, row[j]/sum)
		}
	}
	return result
}
func CrossEntropy(t *mat.Dense, p *mat.Dense) float64 {
	r, c := p.Dims()
	result := 0.0

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result += t.At(i, j) * math.Log(p.At(i, j))
		}
	}

	return -result
}

func Sigmoid(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := m.At(i, j)
			var s float64
			if v > 0 {
				s = 1 / (1 + math.Exp(-v))
			} else {
				s = math.Exp(v) / (1 + math.Exp(v))
			}
			result.Set(i, j, s)
		}
	}
	return result
}

func SigmoidDerivative(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(func(i, j int, v float64) float64 {
		s := 1 / (1 + math.Exp(-v))
		return s * (1 - s)
	}, m)
	return result
}

func Relu(m *mat.Dense) *mat.Dense {
	rows, cols := m.Dims()
	result := mat.NewDense(rows, cols, nil)
	result.Copy(m)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := result.At(i, j)
			if val < 0 {
				result.Set(i, j, 0)
			}
		}
	}

	return result
}

func ReluDerivative(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	deriv := mat.NewDense(rows, cols, nil)

	// Filling derivative : 1 if input > 0, else 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			if val > 0 {
				deriv.Set(i, j, 1)
			} else {
				deriv.Set(i, j, 0)
			}
		}
	}
	return deriv
}
