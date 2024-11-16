#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <wiringPi.h>

// 7-Segment GPIO 설정
#define PA 2
#define PB 4
#define PC 1
#define PD 16
#define PE 15
#define PF 8
#define PG 9
#define PDP 0
char nums[10] = {0xc0, 0xf9, 0xa4, 0xb0, 0x99, 0x92, 0x82, 0xf8, 0x80, 0x90};
char pins[8] = {PA, PB, PC, PD, PE, PF, PG, PDP};

// 7-Segment 초기화 함수
void init_pin() {
    for (int i = 0; i < 8; i++) {
        pinMode(pins[i], OUTPUT);
        digitalWrite(pins[i], HIGH);
    }
}

// 7-Segment 표시 함수
void display_digit(int digit) {
    for (int i = 0; i < 8; i++) {
        digitalWrite(pins[i], (nums[digit] >> i) & 0x1);
    }
}

// TensorFlow Lite 모델 로드 및 초기화
std::unique_ptr<tflite::Interpreter> load_model(const char* model_path) {
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        exit(1);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter" << std::endl;
        exit(1);
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        exit(1);
    }

    return interpreter;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <tflite model path>" << std::endl;
        return 1;
    }

    // 모델 로드
    const char* model_path = argv[1];
    auto interpreter = load_model(model_path);

    // 7-Segment 초기화
    wiringPiSetup();
    init_pin();

    // OpenCV로 카메라 열기
    cv::VideoCapture video(0);
    if (!video.isOpened()) {
        std::cerr << "Unable to get video from the camera!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (video.read(frame)) {
        // 이미지 전처리
        cv::Mat gray, resized;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::resize(gray, resized, cv::Size(28, 28));
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);  // Normalize

        // 모델 입력 설정
        float* input_tensor = interpreter->typed_input_tensor<float>(0);
        memcpy(input_tensor, resized.data, 28 * 28 * sizeof(float));

        // 추론 수행
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Failed to invoke TFLite model" << std::endl;
            continue;
        }

        // 추론 결과 읽기
        float* output_tensor = interpreter->typed_output_tensor<float>(0);
        int predicted_digit = std::max_element(output_tensor, output_tensor + 10) - output_tensor;

        // 결과 출력
        std::cout << "Predicted digit: " << predicted_digit << std::endl;

        // 7-Segment에 표시
        display_digit(predicted_digit);

        // OpenCV 창에 표시
        cv::putText(frame, "Predicted: " + std::to_string(predicted_digit),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Video feed", frame);

        // 'q'를 누르면 종료
        if (cv::waitKey(25) == 'q') {
            break;
        }
    }

    // 자원 해제
    video.release();
    cv::destroyAllWindows();

    return 0;
}