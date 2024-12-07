# Makefile for Alcohol Detection Project

# 컴파일러
CC = g++

# 소스 파일
SRC = project.c

# 출력 파일
TARGET = project

# TensorFlow Lite 헤더 및 라이브러리 경로
TFLITE_INCLUDE = /home/mjm/tensorflow/tensorflow/lite/c
TFLITE_LIB = /home/mjm/tensorflow/tensorflow/lite/tools/make/gen/rpi_armv7l/lib

# WiringPi 라이브러리
WIRINGPI_LIB = -lwiringPi

# 컴파일러 플래그
CFLAGS = -I$(TFLITE_INCLUDE)
LDFLAGS = -L$(TFLITE_LIB) -ltensorflow-lite $(WIRINGPI_LIB) -lm -lpthread -ldl

# 기본 빌드 명령
all: $(TARGET)

# 빌드 규칙
$(TARGET): $(SRC)
   $(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# 깨끗한 빌드 환경을 위한 클린 규칙
clean:
   rm -f $(TARGET)
