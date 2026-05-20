CORE_DIR   ?= ../../core
BUILD_DIR  ?= ../../build-core
ORT_LIB_DIR ?= $(BUILD_DIR)/_deps/onnxruntime-src/lib

CFLAGS  = -I$(CORE_DIR)/include -Wall -Wextra -O2
LDFLAGS = -L$(BUILD_DIR) -lvietasr -Wl,-rpath,$(BUILD_DIR) -Wl,-rpath,$(ORT_LIB_DIR)

.PHONY: all clean run

all: quickstart streaming

quickstart: quickstart.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

streaming: streaming.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run: quickstart
	LD_LIBRARY_PATH=$(BUILD_DIR):$(ORT_LIB_DIR) ./quickstart $(WAV)

clean:
	rm -f quickstart streaming
