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

// 7세그먼트 핀 정의
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
    if (wiringPiSetup() == -1) {
        cerr << "Error: WiringPi setup failed." << endl;
        return 1;
    }
    init_pin();

    unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_filename);
    if (!model) {
        cerr << "Failed to load model." << endl;
        return -1;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter) {
        cerr << "Failed to create interpreter." << endl;
        return -1;
    }

    size_t num_devices;
    unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    assert(num_devices > 0);
    const auto& device = devices.get()[0];
    auto* delegate = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
    interpreter->ModifyGraphWithDelegate(delegate);
    interpreter->AllocateTensors();

    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);

    if (!cap.isOpened()) {
    cerr << "Error: Could not open camera" << endl; 
    }

    while (true) {
        Mat frame;
        cap >> frame;
        
        // 프레임이 비었는지 확인
        if (frame.empty()) {
            cerr << "Error: Captured empty frame" << endl;
            continue;
        }

        int frame_width = frame.cols;
        int frame_height = frame.rows;

        int roi_x = 60;
        int roi_y = 60;
        int roi_width = min(200, frame_width - roi_x);
        int roi_height = min(200, frame_height - roi_y);

        if (roi_width <= 0 || roi_height <= 0) {
            cerr << "Error: ROI size is invalid for current frame size." << endl;
            break;
        }

        Rect roi(roi_x, roi_y, roi_width, roi_height);
        Mat digit = frame(roi);

        // 전처리 및 모델 입력 텐서 채우기
        auto input_tensor = interpreter->typed_input_tensor<float>(0);
        preprocessImage(digit, input_tensor);

        // 추론 실행
        interpreter->Invoke();

        // 출력 결과에서 예측된 숫자 추출
        auto output_tensor = interpreter->typed_output_tensor<float>(0);
        int predicted_label = max_element(output_tensor, output_tensor + 10) - output_tensor;

        // 인식된 숫자를 7세그먼트 디스플레이에 출력
        clear_pin();
        set_pin(predicted_label);

        // 인식된 숫자 및 이미지 출력
        cout << "Predicted Number: " << predicted_label << endl;

        // ROI 박스 및 예측된 숫자 텍스트 표시
        rectangle(frame, roi, Scalar(0, 255, 0), 2);
        putText(frame, "Predicted: " + to_string(predicted_label), Point(roi_x, roi_y - 10), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);

        imshow("Camera Feed", frame);
        imshow("ROI - Digit", digit);

        // 500ms 대기 (0.5초 지연)
        usleep(500000);

        // 'q' 키로 종료
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    edgetpu_free_delegate(delegate);
    destroyAllWindows();
    return 0;
}
