#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <wiringPi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define PA 2
#define PB 4
#define PC 1
#define PD 16
#define PE 15
#define PF 8
#define PG 9
#define PDP 0
// for anode display
char nums[10] = {0xc0, 0xf9, 0xa4, 0xb0, 0x99, 0x92, 0x82, 0xf8, 0x80, 0x90};
// WPi pin numbers
char pins[8] = {PA, PB, PC, PD, PE, PF, PG, PDP};

void clear_pin (){
    int i;
    for (i = 0; i < 8; i++)
      digitalWrite(pins[i], 1);
}

void set_pin (int n){
    int i;
    for (i = 0; i < 8; i++)
      digitalWrite(pins[i], (nums[n] >> i) & 0x1);
}


void init_pin (){
    int i;
    for (i = 0; i < 8; i++)
      pinMode(pins[i], OUTPUT);
}

using namespace std;
using namespace cv;

#define MNIST_INPUT "../mnist_dataset/mnist_images"
#define MNIST_LABEL "../mnist_dataset//mnist_labels"

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}






int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load mnist input images
  vector<vector<float>> input_vector;
  read_Mnist(MNIST_INPUT, input_vector);
  std::cout << "Input MNIST Image" << "\n";
  for(int i=0; i<28; ++i){
    for(int j=0; j<28; ++j){
      printf("%3d ", (int)input_vector[i][j]);
    }
    printf("\n");
  }
  
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  tflite::PrintInterpreterState(interpreter.get());
  auto input_tensor = interpreter->typed_input_tensor<float>(0);
  for(int i=0; i<28; ++i) // image rows
    for(int j=0; j<28; ++j) // image cols
      input_tensor[i * 28 + j] = input_vector[i][j] / 255.0;



  VideoCapture cap(0); // Open default camera (0)
  if (!cap.isOpened()) {
    cerr << "Error: Could not open camera.\n";
    return -1;
  }

  cout << "Press 'q' to quit.\n";


while (true) {
    Mat frame;
    cap >> frame; // Capture a frame
    if (frame.empty()) {
      cerr << "Error: Could not capture frame.\n";
      break;
    }

    // Convert to grayscale
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Resize to 28x28 (MNIST input size)
    Mat resized;
    resize(gray, resized, Size(28, 28));

    // Normalize to 0-1 range
    Mat normalized;
    resized.convertTo(normalized, CV_32F, 1.0 / 255);

    // Copy normalized data to input tensor
    memcpy(input_tensor, normalized.data, 28 * 28 * sizeof(float));

    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    // Read output buffer
    auto output_tensor = interpreter->typed_output_tensor<float>(0);

    // Find the predicted digit
    int predicted_digit = max_element(output_tensor, output_tensor + 10) - output_tensor;
    cout << "Predicted digit: " << predicted_digit << endl;

    // Display the frame with prediction
    putText(frame, "Predicted: " + to_string(predicted_digit), Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    imshow("Digit Recognition", frame);

    // Quit on 'q' key press
    if (waitKey(1) == 'q') {
      break;
    }
  }

  cap.release();
  destroyAllWindows();

  return 0;
}