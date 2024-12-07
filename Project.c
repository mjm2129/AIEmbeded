#include <stdio.h>
#include <math.h>
#include <wiringPi.h>
#include <wiringPiSPI.h>

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

    // 센서 저항 Rs 계산
    Rs = RL * ((VREF - Vout) / Vout);

    ratio = Rs / R0;

    return ratio;
}

void buzzer_alert(int frequency, int duration) {
    for (int i = 0; i < duration; i++) {
        digitalWrite(BUZZER_PIN, HIGH);
        delay(frequency / 2); // 부저 켜기
        digitalWrite(BUZZER_PIN, LOW);
        delay(frequency / 2); // 부저 끄기
    }
}

void led_blink(int frequency, int duration) {
    for (int i = 0; i < duration; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(frequency / 2); // LED 켜기
        digitalWrite(LED_PIN, LOW);
        delay(frequency / 2); // LED 끄기
    }
}

int main() {
    // WiringPi 및 SPI 초기화
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

    pinMode(LED_PIN, OUTPUT);                  // LED 출력 모드 설정
    pinMode(BUZZER_PIN, OUTPUT);               // 부저 출력 모드 설정

    digitalWrite(LED_PIN, LOW);                // LED 초기화
    digitalWrite(BUZZER_PIN, LOW);             // 부저 초기화

    int adcChannel = 0;

    float R0 = 180000.00;                      // 공기 중 R0 값 (직접 설정)
    if (R0 <= 0) {
        printf("R0 Failed\n");
        return 1;
    }

    printf("R0: %.2f Ω\n", R0);

    while (1) {
        int adcValue = read_mcp3208_adc(adcChannel);
        if (adcValue == -1) {
            printf("ADC Read Error\n");
            continue;
        }

        float ratio = calculate_ratio(adcValue, R0);

        printf("ADC Value: %d, Ratio: %.2f\n", adcValue, ratio);

        // Ratio가 1.7 미만일 경우 경고 LED 및 부저 활성화
        if (ratio < 1.7) {
            if (ratio < 1.5) {
                led_blink(100, 10);            // 더 빠르게 깜빡임 (주기 100ms)
                //buzzer_alert(100, 10);        // 더 빠른 경고음 (주기 100ms)
                printf("Critical Warning! Alcohol level is too high!\n");
            } else {
                led_blink(500, 5);            // 느리게 깜빡임 (주기 500ms)
                //buzzer_alert(500, 5);         // 느린 경고음 (주기 500ms)
                printf("Warning! Alcohol detected.\n");
            }
        } else {
            digitalWrite(LED_PIN, LOW);        // LED 끄기
            digitalWrite(BUZZER_PIN, LOW);     // 부저 끄기
        }

        delay(1000); // 1초 간격으로 측정
    }

    return 0;
}
