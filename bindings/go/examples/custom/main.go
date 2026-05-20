package main

import (
	"fmt"
	"os"

	vietasr "github.com/dangvansam/viet-asr/bindings/go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <wav-file>\n", os.Args[0])
		os.Exit(1)
	}

	pipe := vietasr.NewPipeline()
	defer pipe.Close()

	for _, mod := range []string{"vad", "vietasr", "gender", "emotion", "dialect", "noise"} {
		if err := pipe.Add(mod, nil); err != nil {
			fmt.Fprintf(os.Stderr, "add %s: %v\n", mod, err)
			os.Exit(2)
		}
	}
	if err := pipe.Build(); err != nil {
		fmt.Fprintf(os.Stderr, "build: %v\n", err)
		os.Exit(3)
	}

	result, err := pipe.TranscribeFile(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "transcribe: %v\n", err)
		os.Exit(4)
	}
	fmt.Printf("text:    %s...\n", truncate(result.Text, 80))
	fmt.Printf("gender:  %v\n", result.Field("gender"))
	fmt.Printf("emotion: %v\n", result.Field("emotion"))
	fmt.Printf("dialect: %v\n", result.Field("dialect"))
	fmt.Printf("noise:   %v\n", result.Field("noise"))
}

func truncate(s string, n int) string {
	if len([]rune(s)) <= n {
		return s
	}
	return string([]rune(s)[:n])
}
