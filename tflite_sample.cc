#include <opencv2/opencv.hpp>
#include <wiringPi.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include "headers/edgetpu_c.h"
#include <unistd.h>

using namespace cv;
using namespace std;

// 7-Segment 핀 정의
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

// 7-Segment 초기화 및 제어 함수
void clear_pin() {
    for (int i = 0; i < 8; i++) {
        digitalWrite(pins[i], 1);
    }
}

void set_pin(int n) {
    for (int i = 0; i < 8; i++) {
        digitalWrite(pins[i], (nums[n] >> i) & 0x1);
    }
}

void init_pin() {
    for (int i = 0; i < 8; i++) {
        pinMode(pins[i], OUTPUT);
    }
}

// 이미지 전처리 함수 (카메라 프레임 -> 28x28 MNIST 입력 형식)
void preprocessImage(const Mat& src, float* input_tensor) {
    Mat gray, resized;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    resize(gray, resized, Size(28, 28));

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input_tensor[i * 28 + j] = resized.at<uchar>(i, j) / 255.0f;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <tflite model>" << endl;
        return 1;
    }

    const char* model_filename = argv[1];

    // WiringPi 초기화
    if (wiringPiSetup() == -1) {
        cerr << "Error: WiringPi setup failed." << endl;
        return 1;
    }
    init_pin();

    // TensorFlow Lite 모델 로드
    unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_filename);
    if (!model) {
        cerr << "Failed to load model." << endl;
        return -1;
    }

    // Edge TPU 장치 검색 및 연결
    size_t num_devices;
    unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    if (num_devices == 0) {
        cerr << "Error: No Edge TPU devices found." << endl;
        return -1;
    }

    const auto& device = devices.get()[0];
    auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);

    // TFLite 인터프리터 초기화
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter) {
        cerr << "Failed to create interpreter." << endl;
        return -1;
    }

    interpreter->ModifyGraphWithDelegate(delegate);
    interpreter->AllocateTensors();

    // 카메라 초기화
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera." << endl;
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            cerr << "Error: Captured empty frame." << endl;
            continue;
        }

        // ROI 설정 (숫자 영역 추출)
        int roi_x = 60, roi_y = 60, roi_width = 200, roi_height = 200;
        Rect roi(roi_x, roi_y, roi_width, roi_height);
        Mat digit = frame(roi);

        // 이미지 전처리 및 입력 텐서 준비
        auto input_tensor = interpreter->typed_input_tensor<float>(0);
        preprocessImage(digit, input_tensor);

        // 모델 추론
        if (interpreter->Invoke() != kTfLiteOk) {
            cerr << "Error: Model inference failed." << endl;
            continue;
        }

        // 예측 결과 확인
        auto output_tensor = interpreter->typed_output_tensor<float>(0);
        int predicted_label = max_element(output_tensor, output_tensor + 10) - output_tensor;

        // 7-Segment 디스플레이에 출력
        clear_pin();
        set_pin(predicted_label);

        // 결과 출력
        cout << "Predicted Number: " << predicted_label << endl;

        // ROI 및 예측 결과 표시
        rectangle(frame, roi, Scalar(0, 255, 0), 2);
        putText(frame, "Predicted: " + to_string(predicted_label),
                Point(roi_x, roi_y - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);

        imshow("Camera Feed", frame);
        imshow("ROI - Digit", digit);

        // 500ms 대기
        usleep(500000);

        // 'q' 키로 종료
        if (waitKey(1) == 'q') break;
    }

    // 리소스 정리
    cap.release();
    edgetpu_free_delegate(delegate);
    destroyAllWindows();

    return 0;
}
