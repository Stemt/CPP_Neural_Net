CC = g++

LDFLAGS = -lm
CPPFLAGS = --std=c++17 -Iinclude -I/usr/include -MMD -MP -g -O3

BIN_PATH = build/bin
OBJ_PATH = build/obj
SRC_PATH = src

TARGET = $(BIN_PATH)/app

SRC = $(wildcard $(SRC_PATH)/*)
OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC))

.PHONY: all
all: build $(TARGET)

build:
	mkdir build
	mkdir $(BIN_PATH)
	mkdir $(OBJ_PATH)

$(TARGET): $(OBJ) 
	echo $(SRC)
	echo $(OBJ)
	$(CC) $(CPPFLAGS) $(OBJ) $(LDFLAGS) -o $@


clean:
	rm -r build