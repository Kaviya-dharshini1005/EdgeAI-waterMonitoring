#include "autoencoder.h"
#include "weights.h"
#include "normalization.h"
#include "threshold.h"

#include <math.h>

/* ---------------------------------------------------------
 * Helper functions
 * --------------------------------------------------------- */

static float normalize_value(float value, float min, float max)
{
    return (value - min) / (max - min);
}

static float relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

/* ---------------------------------------------------------
 * Autoencoder initialization
 * --------------------------------------------------------- */

void AE_Init(void)
{
    /* No runtime initialization required.
       Model parameters are stored in weights.h. */
}

/* ---------------------------------------------------------
 * Autoencoder prediction
 *
 * Input:
 *   ph
 *   tds
 *   turbidity
 *
 * Output:
 *   Reconstruction error (MSE)
 *
 * Network:
 *   3 -> 8 -> 4 -> 2 -> 4 -> 8 -> 3
 * --------------------------------------------------------- */

float AE_Predict(float ph, float tds, float turbidity)
{
    float input[3];

    float enc1[8];
    float enc2[4];
    float latent[2];

    float dec1[4];
    float dec2[8];
    float output[3];

    int i;
    int j;

    /* -----------------------------------------------------
     * 1. Normalize sensor inputs
     * ----------------------------------------------------- */

    input[0] = normalize_value(ph, PH_MIN, PH_MAX);
    input[1] = normalize_value(tds, TDS_MIN, TDS_MAX);
    input[2] = normalize_value(turbidity,
                               TURBIDITY_MIN,
                               TURBIDITY_MAX);

    /* -----------------------------------------------------
     * 2. Encoder Layer 1
     *    3 -> 8
     * ----------------------------------------------------- */

    for (i = 0; i < 8; i++)
    {
        enc1[i] = ae_enc_b1[i];

        for (j = 0; j < 3; j++)
        {
            enc1[i] += input[j] * ae_enc_w1[j][i];
        }

        enc1[i] = relu(enc1[i]);
    }

    /* -----------------------------------------------------
     * 3. Encoder Layer 2
     *    8 -> 4
     * ----------------------------------------------------- */

    for (i = 0; i < 4; i++)
    {
        enc2[i] = ae_enc_b2[i];

        for (j = 0; j < 8; j++)
        {
            enc2[i] += enc1[j] * ae_enc_w2[j][i];
        }

        enc2[i] = relu(enc2[i]);
    }

    /* -----------------------------------------------------
     * 4. Latent Layer
     *    4 -> 2
     * ----------------------------------------------------- */

    for (i = 0; i < 2; i++)
    {
        latent[i] = ae_enc_b3[i];

        for (j = 0; j < 4; j++)
        {
            latent[i] += enc2[j] * ae_enc_w3[j][i];
        }

        latent[i] = relu(latent[i]);
    }

    /* -----------------------------------------------------
     * 5. Decoder Layer 1
     *    2 -> 4
     * ----------------------------------------------------- */

    for (i = 0; i < 4; i++)
    {
        dec1[i] = ae_dec_b1[i];

        for (j = 0; j < 2; j++)
        {
            dec1[i] += latent[j] * ae_dec_w1[j][i];
        }

        dec1[i] = relu(dec1[i]);
    }

    /* -----------------------------------------------------
     * 6. Decoder Layer 2
     *    4 -> 8
     * ----------------------------------------------------- */

    for (i = 0; i < 8; i++)
    {
        dec2[i] = ae_dec_b2[i];

        for (j = 0; j < 4; j++)
        {
            dec2[i] += dec1[j] * ae_dec_w2[j][i];
        }

        dec2[i] = relu(dec2[i]);
    }

    /* -----------------------------------------------------
     * 7. Output Layer
     *    8 -> 3
     *
     * Output activation is linear.
     * ----------------------------------------------------- */

    for (i = 0; i < 3; i++)
    {
        output[i] = ae_dec_b3[i];

        for (j = 0; j < 8; j++)
        {
            output[i] += dec2[j] * ae_dec_w3[j][i];
        }
    }

    /* -----------------------------------------------------
     * 8. Reconstruction Error
     *
     * MSE = 1/3 * sum((input - output)^2)
     * ----------------------------------------------------- */

    float mse = 0.0f;

    for (i = 0; i < 3; i++)
    {
        float error = input[i] - output[i];
        mse += error * error;
    }

    mse /= 3.0f;

    return mse;
}
