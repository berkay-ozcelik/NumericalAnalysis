import math
import numpy as np

import bitmanip
from bitmanip import BitMap
from numpy import array
from numpy import uint16, uint8
from enum import IntEnum


class Precision(IntEnum):
    binary16 = 0
    binary32 = 1
    binary64 = 2
    binary128 = 3
    binary256 = 4

    def mantissa_bit_width(self) -> int:

        if self == Precision.binary16:
            return 10
        if self == Precision.binary32:
            return 23
        if self == Precision.binary64:
            return 52
        if self == Precision.binary128:
            return 112
        if self == Precision.binary256:
            return 236

    def exponent_bit_width(self):
        if self == Precision.binary16:
            return 5
        if self == Precision.binary32:
            return 8
        if self == Precision.binary64:
            return 11
        if self == Precision.binary128:
            return 15
        if self == Precision.binary256:
            return 19

    def bias(self):
        bias = 2 ** self.exponent_bit_width()
        bias -= 2
        bias //= 2
        return bias

    def total_width(self):
        return 16 * 2 ** self

    def sign_bit_width(self):
        return 1

    def get_numpy_array(self) -> array:
        arr_size = self.total_width() // 16
        return np.zeros(arr_size, dtype=uint16)


class RealNumber:
    __repr: array

    def __init__(self, precision: Precision, sign: int, int_part: int, frac_part: int):

        if int_part == 0 and frac_part == 0:
            self.__repr = precision.get_numpy_array()
            return

        available_mantissa_width = precision.mantissa_bit_width()

        if int_part > 0:
            bitmap_of_integer_part = BitMap(int_part)
            bit_width_of_integer_part = self.shift_left_integer_part_until_nonzero_digit_leads_and_get_bit_width(
                bitmap_of_integer_part)
            available_mantissa_width -= bit_width_of_integer_part
            if available_mantissa_width <= 0:
                exponent = precision.mantissa_bit_width() - available_mantissa_width
                self.write_exponent_part(exponent, precision)
                self.write_integer_part(bitmap_of_integer_part, precision, bit_width_of_integer_part)
            else:
                exponent = bit_width_of_integer_part
                self.write_exponent_part(exponent, precision)
                self.write_integer_part(bitmap_of_integer_part, precision, bit_width_of_integer_part)

                bytes_needed = available_mantissa_width // 8 + 1
                bitmap_of_frac_part = BitMap(np.zeros(bytes_needed, dtype=uint8))
                RealNumber.fill_bitmap_of_frac_part(bitmap_of_frac_part, available_mantissa_width, frac_part)
                self.write_fractional_part(bitmap_of_frac_part, precision, bit_width_of_integer_part,
                                           available_mantissa_width)
        else:
            bytes_needed = available_mantissa_width // 8 + 1
            bitmap_of_frac_part = BitMap(np.zeros(bytes_needed, dtype=uint8))
            RealNumber.fill_bitmap_of_frac_part(bitmap_of_frac_part, available_mantissa_width, frac_part)
            shift_amount = self.shift_left_fractional_part_until_nonzero_digit_leads_and_get_shift_amount(
                bitmap_of_frac_part)
            exponent = -shift_amount
            self.write_exponent_part(exponent, precision)
            self.write_fractional_part(bitmap_of_frac_part, precision, 0, available_mantissa_width)

        self.write_sign_bit(sign)

    def shift_left_fractional_part_until_nonzero_digit_leads_and_get_shift_amount(self, bitmap_of_fractional_part):
        shift_amount = bitmap_of_fractional_part.first_index_of(1) + 1
        bitmap_of_fractional_part << shift_amount
        return shift_amount

    def shift_left_integer_part_until_nonzero_digit_leads_and_get_bit_width(self, bitmap_of_integer_part):
        begin_offset_of_integer_part = bitmap_of_integer_part.first_index_of(1) + 1
        bit_width_of_integer_part = len(bitmap_of_integer_part) - begin_offset_of_integer_part
        bitmap_of_integer_part << begin_offset_of_integer_part
        return bit_width_of_integer_part

    def write_sign_bit(self, sign):
        bitmap = BitMap(self.__repr)
        bitmap[0] = sign

    def write_fractional_part(self, bitmap_of_fractional_part, precision, bit_width_of_integer_part,
                              available_mantissa_bit_width):
        base_index = bit_width_of_integer_part + precision.exponent_bit_width() + 1
        bitmap = BitMap(self.__repr)
        for i in range(available_mantissa_bit_width):
            bitmap[i + base_index] = bitmap_of_fractional_part[i]

    def write_integer_part(self, bitmap_of_integer_part, precision, bit_width_of_integer_part):
        base_index = precision.exponent_bit_width() + 1
        size = min(bit_width_of_integer_part, precision.mantissa_bit_width())

        bitmap = BitMap(self.__repr)

        for i in range(size):
            bitmap[i + base_index] = bitmap_of_integer_part[i]

    def write_exponent_part(self, exponent, precision):
        biased_exponent = exponent + precision.bias()
        bitmap_of_biased_exponent = BitMap(biased_exponent)
        begin_offset_of_exponent_part = len(bitmap_of_biased_exponent) - precision.exponent_bit_width()
        bitmap_of_biased_exponent << begin_offset_of_exponent_part
        self.__repr = precision.get_numpy_array()
        bitmap = BitMap(self.__repr)
        for i in range(precision.exponent_bit_width()):
            bitmap[i + 1] = bitmap_of_biased_exponent[i]

    def value(self):
        return self.__repr

    @staticmethod
    def fill_bitmap_of_frac_part(bitmap: BitMap, mantissa_width: int, frac_part: int):
        for i in range(mantissa_width):
            pre = int(math.log10(frac_part))
            frac_part *= 2
            post = int(math.log10(frac_part))
            if post - pre:
                bitmap[i] = post - pre
                frac_part -= 10 ** post
            if frac_part == 0:
                break

