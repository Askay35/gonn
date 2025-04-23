package math

import (
	"github.com/Askay35/gonn/utils"
	"gonum.org/v1/gonum/mat"
)

func RandomDense(r, c int, min, max float64) *mat.Dense {
	dense := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dense.Set(i, j, utils.RandFloat64(min, max))
		}
	}
	return dense
}

func AddVectorToMatrix(m *mat.Dense, b *mat.Dense) *mat.Dense {
	// Dimensions check
	mr, mc := m.Dims()
	br, bc := b.Dims()

	if br != 1 || bc != mc {
		panic("b.rows must = 1 and b.columns = m.columns")
	}

	result := mat.NewDense(mr, mc, nil)
	result.Copy(m)

	// Add b to every row of a
	for i := 0; i < mr; i++ {
		for j := 0; j < mc; j++ {
			result.Set(i, j, result.At(i, j)+b.At(0, j))
		}
	}

	return result
}

func MultiplyMatrix(a *mat.Dense, b *mat.Dense) *mat.Dense {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic("wrong matrix sizes")
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
