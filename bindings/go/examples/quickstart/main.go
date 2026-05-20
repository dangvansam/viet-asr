package main

import (
	"fmt"
	"os"

	vietasr "github.com/dangvansam/viet-asr/bindings/go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <wav-file>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\npresets: %v\n", vietasr.ListPresets())
		fmt.Fprintf(os.Stderr, "modules: %v\n", vietasr.ListModules())
		os.Exit(1)
	}

	pipe, err := vietasr.PipelinePreset("transcribe")
	if err != nil {
		fmt.Fprintf(os.Stderr, "preset: %v\n", err)
		os.Exit(2)
	}
	defer pipe.Close()

	result, err := pipe.TranscribeFile(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "transcribe: %v\n", err)
		os.Exit(3)
	}
	fmt.Println(result.Text)
}
