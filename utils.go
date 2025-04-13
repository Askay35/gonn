package main

import (
	"math"
	"math/rand"

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

func randomDense(r, c int, min, max float64) *mat.Dense {
	dense := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dense.Set(i, j, randFloat64(min, max))
		}
	}
	return dense
}

func randFloat64(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func addVectorToMatrix(m *mat.Dense, b *mat.Dense) *mat.Dense {
	// Проверка размеров
	mr, mc := m.Dims()
	br, bc := b.Dims()

	if br != 1 || bc != mc {
		panic("b должен быть вектором-строкой, совпадающим по размеру с числом столбцов m")
	}

	result := mat.NewDense(mr, mc, nil)
	result.Copy(m)

	// Добавляем b к каждой строке a
	for i := 0; i < mr; i++ {
		for j := 0; j < mc; j++ {
			result.Set(i, j, result.At(i, j)+b.At(0, j))
		}
	}

	return result
}

func getAnswer(o *mat.Dense) int {
	if o.At(0, 0) > o.At(0, 1) {
		return 0
	}
	return 1
}

func manualMultiply(a *mat.Dense, b *mat.Dense) *mat.Dense {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic("несовместимые размеры матриц")
	}

	res := mat.NewDense(ar, bc, nil)
	for i := 0; i < ar; i++ {
		for j := 0; j < bc; j++ {
			sum := 0.0
			for k := 0; k < ac; k++ {
				sum += a.At(i, k) * b.At(k, j)
			}
			res.Set(i, j, sum)
		}
	}
	return res
}
