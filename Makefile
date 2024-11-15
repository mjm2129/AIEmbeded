# 컴파일러 설정
CC := g++
CFLAGS := -std=c++11 -I/usr/include

# include 경로
INCLUDEPATHS := -I${HOME}/tensorflow -I${HOME}/EAI

# 링커 플래그에 wiringPi 및 OpenCV 추가
LDFLAGS := -pthread -ltensorflow-lite -lflatbuffers -l:libedgetpu.so.1.0 -ldl -lwiringPi $(shell pkg-config --libs opencv4)

# 라이브러리 경로 설정
LDPATH := -L${HOME}/tensorflow/tensorflow/lite/tools/make/gen/bbb_armv7l/lib \
          -L${HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build \
          -L${HOME}/EAI/libs/armv7a

# 소스 파일 및 빌드 대상
SRCS := tflite_sample.cc
OBJS := $(SRCS:.cc=.o)
EXEC := tflite_sample

# 빌드 규칙
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -o $@ $^ $(LDPATH) $(LDFLAGS)

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDEPATHS) $(shell pkg-config --cflags opencv4) -c $< -o $@

# 클린 규칙
clean:
	rm -f $(OBJS) $(EXEC)
