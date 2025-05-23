package utils

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

type MNISTData struct {
	Pixels                   []byte
	Labels                   []byte
	PixelsSize, ImagesNumber int
}

const (
	IMAGE_SIZE_BYTES = 784
	IMAGE_WIDTH      = 28
	IMAGE_HEIGHT     = 28
)

func ReadMNISTImages(images_path string) ([]byte, int, int, int, error) {
	images_file, err := os.Open(images_path)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	defer images_file.Close()

	images_reader := io.Reader(images_file)

	// Reading mnist header (magic number, images number, rows, cols)
	var magic, temp_num_images, temp_rows, temp_cols int32
	err = binary.Read(images_reader, binary.BigEndian, &magic)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	err = binary.Read(images_reader, binary.BigEndian, &temp_num_images)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	err = binary.Read(images_reader, binary.BigEndian, &temp_rows)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	err = binary.Read(images_reader, binary.BigEndian, &temp_cols)
	if err != nil {
		return nil, 0, 0, 0, err
	}

	num_images, rows, cols := int(temp_num_images), int(temp_rows), int(temp_cols)
	data := make([]byte, num_images*rows*cols)
	_, err = images_reader.Read(data)
	if err != nil {
		return nil, 0, 0, 0, err
	}

	return data, num_images, rows, cols, nil
}

func ReadMNISTLabels(labels_path string) ([]byte, error) {
	labels_file, err := os.Open(labels_path)
	if err != nil {
		return nil, err
	}
	defer labels_file.Close()

	labels_reader := io.Reader(labels_file)

	var magic_number, size int32
	err = binary.Read(labels_reader, binary.BigEndian, &magic_number)
	if err != nil {
		return nil, err
	}
	err = binary.Read(labels_reader, binary.BigEndian, &size)
	if err != nil {
		return nil, err
	}

	labels := make([]byte, size)
	_, err = labels_reader.Read(labels)
	if err != nil {
		return nil, err
	}

	return labels, nil
}

func ReadMNISTData(images_path, labels_path string) (MNISTData, error) {
	pixels, num_images, rows, cols, err := ReadMNISTImages(images_path)
	if err != nil {
		return MNISTData{}, err
	}
	labels, err := ReadMNISTLabels(labels_path)
	if err != nil {
		return MNISTData{}, err
	}

	data := MNISTData{}
	data.ImagesNumber = num_images
	data.PixelsSize = num_images * rows * cols
	data.Pixels = pixels
	data.Labels = labels

	// fmt.Printf("Loaded %d images. Size %dx%d\n", data.ImagesNumber, IMAGE_WIDTH, IMAGE_HEIGHT)
	// fmt.Printf("Loaded %d labels\n", len(data.Labels))

	return data, nil
}

func PrintMNIST(data []float64) {
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
