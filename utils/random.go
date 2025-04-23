package utils

import "math/rand"

func RandFloat64(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
