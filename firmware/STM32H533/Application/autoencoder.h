#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <stdint.h>

/* Initialize Autoencoder */
void AE_Init(void);

/* Predict anomaly score */
float AE_Predict(float ph, float tds, float turbidity);

#endif
