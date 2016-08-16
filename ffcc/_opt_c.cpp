// Copyright (c) BlackFurniture.
// See LICENSE for details.

#include <stdint.h>

void DecodeADPCM(uint8_t *src, int16_t *dst,
                 size_t num_nibbles, size_t nibble_pos,
                 DSPHeader * d, int channels)
{
    while (nibble_pos < num_nibbles) {
        if ((nibble_pos & 15) == 0) {
            d->pred_scale = src[(nibble_pos & ~15) >> 1];
            nibble_pos += 2;
        }
        if (nibble_pos >= num_nibbles)
            break;
        int scale = 1 << (d->pred_scale & 0xF);
        int coef_index = (d->pred_scale >> 4);
        int32_t coef1 = d->coefs[coef_index][0];
        int32_t coef2 = d->coefs[coef_index][1];

        uint8_t byte = src[nibble_pos >> 1];
        int temp = (nibble_pos & 1) ? (byte & 0xF) : (byte >> 4);
        if (temp >= 8)
            temp -= 16;

        int val = (scale * temp) +
                  ((0x400 + coef1 * d->yn1 + coef2 * d->yn2) >> 11);
        if (val > 0x7FFF)
            val = 0x7FFF;
        else if (val < -0x7FFF)
            val = -0x7FFF;

        d->yn2 = d->yn1;
        d->yn1 = val;
        nibble_pos += 1;
        *dst = val;
        dst += channels;
    }
}
