package main

import (
	"encoding/binary"
	"io"
	"os"
)

type MNISTData struct {
	pixels                     []byte
	labels                     []byte
	pixels_size, images_number int
}

const (
	IMAGE_SIZE_BYTES = 784
	IMAGE_WIDTH      = 28
	IMAGE_HEIGHT     = 28
)

func readMNISTImages(images_path string) ([]byte, int, int, int, error) {
	images_file, err := os.Open(images_path)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	defer images_file.Close()

	images_reader := io.Reader(images_file)

	// Чтение заголовка (magic number, количество изображений, rows, cols)
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

func readMNISTLabels(labels_path string) ([]byte, error) {
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

func readMNISTData(images_path, labels_path string) (MNISTData, error) {
	pixels, num_images, rows, cols, err := readMNISTImages(images_path)
	if err != nil {
		return MNISTData{}, err
	}
	labels, err := readMNISTLabels(labels_path)
	if err != nil {
		return MNISTData{}, err
	}

	data := MNISTData{}
	data.images_number = num_images
	data.pixels_size = num_images * rows * cols
	data.pixels = pixels
	data.labels = labels

	return data, nil
}
