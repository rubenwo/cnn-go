package mnist

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/rubenwo/cnn-go/pkg/cnn/maths"
	"io"
	"os"
)

func ReadGrayImages(path string, limit int) ([]maths.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("couldn't open file: %w", err)
	}
	defer f.Close()

	reader := bufio.NewReader(f)

	var header struct{ Magic, N, Rows, Cols int32 }
	if err := binary.Read(reader, binary.BigEndian, &header); err != nil {
		return nil, errors.New("bad header")
	}
	if header.Magic != 2051 {
		return nil, errors.New("wrong magic number in header")
	}
	bytes := make([]byte, header.N*header.Rows*header.Cols)
	if _, err = io.ReadFull(reader, bytes); err != nil {
		return nil, fmt.Errorf("%w, could not read full", err)
	}

	if limit > int(header.N) {
		return nil, fmt.Errorf("limit is larger than the amount of images in the dataset")
	}

	byteIndex := 0

	images := make([]maths.Tensor, limit)
	for i := 0; i < limit; i++ {
		pixels := make([]float64, header.Cols*header.Rows)
		for j := 0; j < len(pixels); j++ {
			pixels[j] = (128.0 - float64(bytes[byteIndex])) / 255.0
			byteIndex++
		}

		images[i] = *maths.NewTensor([]int{int(header.Cols), int(header.Rows)}, pixels)
	}

	return images, nil
}

func ReadLabels(path string, limit int) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("couldn't open file: %w", err)
	}
	defer f.Close()
	reader := bufio.NewReader(f)
	var header struct{ Magic, N int32 }
	if err := binary.Read(reader, binary.BigEndian, &header); err != nil {
		return nil, errors.New("bad header")
	}
	if header.Magic != 2049 {
		return nil, errors.New("wrong magic number in header")
	}

	bytes := make([]byte, header.N)
	if _, err = io.ReadFull(reader, bytes); err != nil {
		return nil, fmt.Errorf("%w, could not read full", err)
	}
	if limit > int(header.N) {
		return nil, fmt.Errorf("limit is larger than the amount of labels in the dataset")
	}

	labels := make([]int, limit)
	for i := 0; i < limit; i++ {
		labels[i] = int(bytes[i])
	}

	return labels, nil
}

func LabelsToTensors(labels []int) []maths.Tensor {
	tensors := make([]maths.Tensor, len(labels))
	for i := 0; i < len(labels); i++ {
		values := make([]float64, 10)
		values[labels[i]] = 1
		t := maths.NewTensor([]int{10}, values) // 1d tensor of 10 values for 0-9
		tensors[i] = *t
	}
	return tensors
}
