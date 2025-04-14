package main

import (
	"math/rand"
)

func randFloat64(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
func bytesToFloats(data []byte) []float64 {
	floats := make([]float64, len(data))
	for i, b := range data {
		floats[i] = float64(b)
	}
	return floats
}
func labelsToOuputs(labels []byte, size int) []float64 {
	var outputs []float64
	for _, b := range labels {
		for j := 0; j < size; j++ {
			if j == int(b) {
				outputs = append(outputs, 1.0)
			} else {
				outputs = append(outputs, 0.0)
			}
		}
	}
	return outputs
}
