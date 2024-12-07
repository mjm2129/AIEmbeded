#include <stdio.h>
#include <math.h>
#include <wiringPi.h>
#include <wiringPiSPI.h>
// TensorFlow Lite C API 헤더 경로
#include "common.h"
#include "c_api.h"

#define SPI_CHANNEL 0        // SPI 채널
#define CS_MCP3208 8         // MCP3208의 CS 핀 GPIO 번호
#define RL 100000.0          // MQ-3 로드 저항 (100kΩ)
#define VREF 3.3             // 참조 전압 (3.3V)

#define LED_PIN 1            // WiringPi 기준 GPIO 1 -> 물리적 핀 12 (GPIO 18)
#define BUZZER_PIN 6         // WiringPi 기준 GPIO 6 -> 물리적 핀 22 (GPIO 25)

int read_mcp3208_adc(unsigned char adcChannel) {
    unsigned char buff[3];
    int adcValue = 0;

    buff[0] = 0x06 | ((adcChannel & 0x07) >> 2);
    buff[1] = (adcChannel & 0x07) << 6;
    buff[2] = 0x00;

    digitalWrite(CS_MCP3208, LOW);
    wiringPiSPIDataRW(SPI_CHANNEL, buff, 3);
    digitalWrite(CS_MCP3208, HIGH);

    adcValue = ((buff[1] & 0x0F) << 8) | buff[2];

    return adcValue;
}

float calculate_ratio(int adcValue, float R0) {
    float Vout = 0.0, Rs = 0.0, ratio = 0.0;

    Vout = (adcValue / 4095.0) * VREF;

    Rs = RL * ((VREF - Vout) / Vout);
    ratio = Rs / R0;

    return ratio;
}

void buzzer_alert(int frequency, int duration) {
    for (int i = 0; i < duration; i++) {
        digitalWrite(BUZZER_PIN, HIGH);
        delay(frequency / 2);
        digitalWrite(BUZZER_PIN, LOW);
        delay(frequency / 2);
    }
}

void led_blink(int frequency, int duration) {
    for (int i = 0; i < duration; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(frequency / 2);
        digitalWrite(LED_PIN, LOW);
        delay(frequency / 2);
    }
}

float predict_with_lstm(float ratios[8], TfLiteInterpreter *interpreter, TfLiteTensor *input_tensor) {
    // 입력 텐서에 데이터를 복사
    for (int i = 0; i < 8; i++) {
        input_tensor->data.f[i] = ratios[i];
    }

    // 모델 실행
    if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
        fprintf(stderr, "Error invoking the TensorFlow Lite interpreter\n");
        return -1;
    }

    // 출력 텐서 가져오기
    const TfLiteTensor *output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    if (output_tensor == NULL) {
        fprintf(stderr, "Failed to get output tensor\n");
        return -1;
    }

    // 출력 데이터 가져오기
    const float *output_data = (const float *)TfLiteTensorData(output_tensor);

    if (output_data == NULL) {
        fprintf(stderr, "Failed to get output data\n");
        return -1;
    }

    // 첫 번째 출력 값을 반환
    return output_data[0];
}

void print_ratios(float ratios[], int size) {
    printf("Current Ratios: [");
    for (int i = 0; i < size; i++) {
        printf("%.2f", ratios[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main() {
    if (wiringPiSetup() == -1) {
        printf("WiringPi Error\n");
        return 1;
    }

    if (wiringPiSPISetup(SPI_CHANNEL, 1000000) == -1) {
        printf("SPI Error\n");
        return 1;
    }

    pinMode(CS_MCP3208, OUTPUT);
    digitalWrite(CS_MCP3208, HIGH);

    pinMode(LED_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);

    digitalWrite(LED_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);

    float R0 = 180000.0;
    if (R0 <= 0) {
        printf("R0 Failed\n");
        return 1;
    }

    // TensorFlow Lite 모델 로드
    TfLiteModel *model = TfLiteModelCreateFromFile("lstm_drink_model.tflite");
    if (!model) {
        fprintf(stderr, "Failed to load TFLite model\n");
        return 1;
    }

    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    if (!interpreter) {
        fprintf(stderr, "Failed to create interpreter\n");
        return 1;
    }

    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        fprintf(stderr, "Failed to allocate tensors\n");
        return 1;
    }

    TfLiteTensor *input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    if (!input_tensor) {
        fprintf(stderr, "Failed to get input tensor\n");
        return 1;
    }

    float ratios[8] = {0.0}; // 최근 측정된 8개의 비율 저장
    int adcChannel = 0;

    printf("R0: %.2f Ω\n", R0);

    while (1) {
        for (int i = 0; i < 8; i++) {
            int adcValue = read_mcp3208_adc(adcChannel);
            ratios[i] = calculate_ratio(adcValue, R0);
            delay(500); // 500ms 간격으로 측정
        }

        print_ratios(ratios, 8); // 현재 ratio 값 출력

        float probability = predict_with_lstm(ratios, interpreter, input_tensor);
    
        if (probability > 0.5) {
            led_blink(100, 10); // 빠르게 깜빡임
            buzzer_alert(100, 10); // 빠른 경고음
            printf("Critical Warning! Alcohol level detected!\n");
        } else {
            digitalWrite(LED_PIN, LOW);
            digitalWrite(BUZZER_PIN, LOW);
            printf("No alcohol detected.\n");
        }

        delay(1000);
    }

    TfLiteInterpreterDelete(interpreter);
    return 0;
}
