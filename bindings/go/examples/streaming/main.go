package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"time"

	vietasr "github.com/dangvansam/viet-asr/bindings/go"
)

type wavInfo struct {
	channels      uint16
	sampleRate    uint32
	bitsPerSample uint16
}

func readWavInt16Mono(path string) ([]int16, uint32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()

	header := make([]byte, 12)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, 0, err
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("not a RIFF/WAVE file")
	}

	var info wavInfo
	chunkHeader := make([]byte, 8)
	for {
		if _, err := io.ReadFull(f, chunkHeader); err != nil {
			return nil, 0, err
		}
		id := string(chunkHeader[0:4])
		size := binary.LittleEndian.Uint32(chunkHeader[4:8])
		body := make([]byte, size)
		if _, err := io.ReadFull(f, body); err != nil {
			return nil, 0, err
		}
		if id == "fmt " {
			if size < 16 {
				return nil, 0, fmt.Errorf("fmt chunk too small")
			}
			info.channels = binary.LittleEndian.Uint16(body[2:4])
			info.sampleRate = binary.LittleEndian.Uint32(body[4:8])
			info.bitsPerSample = binary.LittleEndian.Uint16(body[14:16])
		}
		if id == "data" {
			if info.bitsPerSample != 16 {
				return nil, 0, fmt.Errorf("only 16-bit PCM supported")
			}
			samples := make([]int16, len(body)/2)
			for i := range samples {
				samples[i] = int16(binary.LittleEndian.Uint16(body[i*2 : i*2+2]))
			}
			if info.channels == 1 {
				return samples, info.sampleRate, nil
			}
			mono := make([]int16, len(samples)/int(info.channels))
			for i := range mono {
				var mixed int32
				for c := 0; c < int(info.channels); c++ {
					mixed += int32(samples[i*int(info.channels)+c])
				}
				mono[i] = int16(mixed / int32(info.channels))
			}
			return mono, info.sampleRate, nil
		}
		if size%2 == 1 {
			extra := make([]byte, 1)
			f.Read(extra)
		}
	}
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <wav-file>\n", os.Args[0])
		os.Exit(1)
	}

	pcm, sampleRate, err := readWavInt16Mono(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(2)
	}
	durationS := float64(len(pcm)) / float64(sampleRate)
	fmt.Printf("audio: %.2fs @ %d Hz\n", durationS, sampleRate)

	pipe, err := vietasr.PipelinePreset("transcribe")
	if err != nil {
		fmt.Fprintf(os.Stderr, "preset: %v\n", err)
		os.Exit(3)
	}
	defer pipe.Close()

	session, err := pipe.Stream(float32(sampleRate))
	if err != nil {
		fmt.Fprintf(os.Stderr, "stream: %v\n", err)
		os.Exit(4)
	}
	defer session.Close()

	chunkSize := int(sampleRate / 1000 * 320)
	started := time.Now()
	lastPartial := ""

	for offset := 0; offset < len(pcm); offset += chunkSize {
		end := offset + chunkSize
		if end > len(pcm) {
			end = len(pcm)
		}
		session.Accept(pcm[offset:end])
		partial := session.Partial().Text
		if partial != "" && partial != lastPartial {
			t := float64(end) / float64(sampleRate)
			fmt.Printf("  [%5.2fs] %s\n", t, partial)
			lastPartial = partial
		}
	}
	final := session.Final().Text
	elapsed := time.Since(started).Seconds()
	fmt.Printf("\nFINAL (%.2fs wall, RTF %.2f):\n%s\n",
		elapsed, elapsed/durationS, final)
}
