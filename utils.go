package main

import (
	"math/rand"
	"os"
	"os/exec"
	"runtime"
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
func clearConsole() {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "linux", "darwin":
		cmd = exec.Command("clear")
	case "windows":
		cmd = exec.Command("cmd", "/c", "cls")
	default:
		return
	}
	cmd.Stdout = os.Stdout
	cmd.Run()
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
