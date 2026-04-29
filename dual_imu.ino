#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include "SparkFun_BNO080_Arduino_Library.h"

#ifndef REPORT_PERIOD_MS
#define REPORT_PERIOD_MS 50U
#endif

#ifndef VALUE_DECIMALS
#define VALUE_DECIMALS 6
#endif

#ifndef REQUIRE_NEW_SAMPLE_FOR_PRINT
#define REQUIRE_NEW_SAMPLE_FOR_PRINT 1
#endif

BNO080 myIMU1;
BNO080 myIMU2;

struct ImuState {
  uint32_t lastMcuMs = 0;
  uint32_t lastBnoUs = 0;
  bool newAny = false;

  float qw = 0, qx = 0, qy = 0, qz = 0;
  uint8_t quatAcc = 0;

  float gvx = 0, gvy = 0, gvz = 0;
  uint8_t gravAcc = 0;

  float gx = 0, gy = 0, gz = 0;
  uint8_t gyroAcc = 0;

  float lax = 0, lay = 0, laz = 0;
  float ax = 0, ay = 0, az = 0;
  float mx = 0, my = 0, mz = 0;
};

static void enableReports(BNO080 &imu) {
  imu.enableRotationVector(REPORT_PERIOD_MS);
  imu.enableGravity(REPORT_PERIOD_MS);
  imu.enableGyro(REPORT_PERIOD_MS);
  imu.enableLinearAccelerometer(REPORT_PERIOD_MS);
  imu.enableAccelerometer(REPORT_PERIOD_MS);
  imu.enableMagnetometer(REPORT_PERIOD_MS);
}

static void drainImu(BNO080 &imu, ImuState &s) {
  s.newAny = false;

  uint16_t r;
  while ((r = imu.getReadings()) != 0) {
    s.lastMcuMs = millis();
    s.lastBnoUs = imu.getTimeStamp();
    s.newAny = true;

    if (r == SENSOR_REPORTID_ROTATION_VECTOR) {
      s.qw = imu.getQuatReal();
      s.qx = imu.getQuatI();
      s.qy = imu.getQuatJ();
      s.qz = imu.getQuatK();
      s.quatAcc = imu.getQuatAccuracy();

    } else if (r == SENSOR_REPORTID_GRAVITY) {
      s.gvx = imu.getGravityX();
      s.gvy = imu.getGravityY();
      s.gvz = imu.getGravityZ();
      s.gravAcc = imu.getGravityAccuracy();

    } else if (r == SENSOR_REPORTID_GYROSCOPE) {
      s.gx = imu.getGyroX();
      s.gy = imu.getGyroY();
      s.gz = imu.getGyroZ();
      s.gyroAcc = imu.getGyroAccuracy();

    } else if (r == SENSOR_REPORTID_LINEAR_ACCELERATION) {
      s.lax = imu.getLinAccelX();
      s.lay = imu.getLinAccelY();
      s.laz = imu.getLinAccelZ();

    } else if (r == SENSOR_REPORTID_ACCELEROMETER) {
      s.ax = imu.getAccelX();
      s.ay = imu.getAccelY();
      s.az = imu.getAccelZ();

    } else if (r == SENSOR_REPORTID_MAGNETIC_FIELD) {
      s.mx = imu.getMagX();
      s.my = imu.getMagY();
      s.mz = imu.getMagZ();
    }
  }
}

static void printComma() {
  Serial.print(",");
}

static void printFloat(float v) {
  Serial.print(v, VALUE_DECIMALS);
}

static void printImu(const ImuState &s) {
  Serial.print(s.lastMcuMs);
  printComma();
  Serial.print(s.lastBnoUs);
  printComma();
  Serial.print(s.newAny ? 1 : 0);
  printComma();

  printFloat(s.qw);
  printComma();
  printFloat(s.qx);
  printComma();
  printFloat(s.qy);
  printComma();
  printFloat(s.qz);
  printComma();

  Serial.print(s.quatAcc);
  printComma();
  Serial.print(s.gravAcc);
  printComma();
  Serial.print(s.gyroAcc);
  printComma();

  printFloat(s.gvx);
  printComma();
  printFloat(s.gvy);
  printComma();
  printFloat(s.gvz);
  printComma();

  printFloat(s.gx);
  printComma();
  printFloat(s.gy);
  printComma();
  printFloat(s.gz);
  printComma();

  printFloat(s.lax);
  printComma();
  printFloat(s.lay);
  printComma();
  printFloat(s.laz);
  printComma();

  printFloat(s.ax);
  printComma();
  printFloat(s.ay);
  printComma();
  printFloat(s.az);
  printComma();

  printFloat(s.mx);
  printComma();
  printFloat(s.my);
  printComma();
  printFloat(s.mz);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000);

  Serial.println("\n=== Dual BNO085 Full IMU Output ===");

  Wire1.begin();
  Wire1.setSCL(16);
  Wire1.setSDA(17);
  Wire1.setClock(400000);

  Wire.begin();
  Wire.setSCL(19);
  Wire.setSDA(18);
  Wire.setClock(400000);

  delay(1000);

  Serial.print("Initializing S1 (16,17 / 0x4B)... ");
  if (!myIMU1.begin(0x4B, Wire1)) {
    Serial.println("Fail!");
  } else {
    enableReports(myIMU1);
    Serial.println("OK!");
  }

  Serial.print("Initializing S2 (19,18 / 0x4B)... ");
  if (!myIMU2.begin(0x4B, Wire)) {
    Serial.println("Fail!");
  } else {
    enableReports(myIMU2);
    Serial.println("OK!");
  }

  Serial.println("tx_ms,seq,t1_mcu_ms,bno1_us,new1,qw1,qx1,qy1,qz1,quatAcc1,gravAcc1,gyroAcc1,gvx1,gvy1,gvz1,gx1,gy1,gz1,lax1,lay1,laz1,ax1,ay1,az1,mx1,my1,mz1,t2_mcu_ms,bno2_us,new2,qw2,qx2,qy2,qz2,quatAcc2,gravAcc2,gyroAcc2,gvx2,gvy2,gvz2,gx2,gy2,gz2,lax2,lay2,laz2,ax2,ay2,az2,mx2,my2,mz2");
}

void loop() {
  static ImuState s1;
  static ImuState s2;
  static uint32_t seq = 0;

  drainImu(myIMU1, s1);
  drainImu(myIMU2, s2);

#if REQUIRE_NEW_SAMPLE_FOR_PRINT
  if (!s1.newAny && !s2.newAny) {
    return;
  }
#else
  static uint32_t lastPrint = 0;
  const uint32_t now = millis();
  if (now - lastPrint < REPORT_PERIOD_MS) {
    return;
  }
  lastPrint = now;
#endif

  seq++;
  Serial.print(millis());
  printComma();
  Serial.print(seq);
  printComma();

  printImu(s1);
  printComma();
  printImu(s2);

  Serial.println();
}
