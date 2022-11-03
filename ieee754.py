import math
import numpy as np
from bitmanip import BitMap
from numpy import array
from numpy import uint16
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

    def __init__(self, precision: Precision, str_repr: str):
        sign: int
        int_part: int
        frac_part: int
        int_part, frac_part = RealNumber.__get_parts_of_str_repr(str_repr)

        int_part_binary_repr: list[uint16]
        frc_part_binary_repr: list[uint16]

        bit_width_of_int_part, int_part_binary_repr = RealNumber.__convert_integer_part_to_binary(int_part)

        available_mantissa_width = precision.mantissa_bit_width() - bit_width_of_int_part

        if available_mantissa_width < 0:
            pass  # TODO:Overflow

        bit_width_of_frc_part, frc_part_binary_repr = RealNumber.__convert_fractional_part_to_binary(frac_part,
                                                                                                     available_mantissa_width)

        # Integer part exists.Just concat two parts, exponent will be one less than width of integer part.
        if int_part > 0:
            exponent = bit_width_of_int_part - 1
        # Integer part not exists but decimal part exists.Shift until first digit is non-zero.
        # Exponent will be -shift_amount.
        elif frac_part > 0:
            shift_amount = RealNumber.__get_nonzero_digit_shift_amount(frc_part_binary_repr)
            RealNumber.__shift_bits_left(frc_part_binary_repr, shift_amount)
            exponent = -shift_amount
        # Neither integer or decimal parts exists, number is zero.
        else:
            self.__repr = precision.get_numpy_array()
            return

        self.__repr = precision.get_numpy_array()


        biased_exponent = exponent + precision.bias()

        #TODO : Copy them all to the final destination.

        pass


    @staticmethod
    def __shift_bits_left(binary_repr: list[uint16], shift_amount: int):
        if shift_amount > 16:
            n = shift_amount // 16
            for i in range(n):
                RealNumber.__shift_bits_left_with_limit_16(binary_repr, 16)
            shift_amount %= 16

        RealNumber.__shift_bits_left_with_limit_16(binary_repr, shift_amount)

    @staticmethod
    def __shift_bits_left_with_limit_16(binary_repr: list[uint16], shift_amount: int) -> uint16:
        if shift_amount > 16:
            raise Exception("Not yet implemented.")

        mask = uint16(0x0000)
        for i in range(shift_amount):
            mask = mask | uint16(0x8000) >> i

        prev_overflow = uint16(0x0000)
        for i in range(len(binary_repr) - 1, -1, -1):
            current = binary_repr[i]
            bits_where_going_to_overflow = current & mask
            current = current << shift_amount
            prev_overflow = prev_overflow >> (16 - shift_amount)
            binary_repr[i] = current | prev_overflow
            prev_overflow = bits_where_going_to_overflow
        return prev_overflow

    @staticmethod
    def __get_nonzero_digit_shift_amount(binary_repr_of_fraction: list[uint16]) -> int:
        mask = 0x8000
        count = 1
        for bits_on_uint16 in binary_repr_of_fraction:
            for i in range(16):
                is_bit_nonzero = mask & bits_on_uint16
                if is_bit_nonzero:
                    return count
                count += 1
                bits_on_uint16 = bits_on_uint16 << 1

    @staticmethod
    def __flush_digit_list_for_int_part(list_of_digits: [int]) -> uint16:
        bits_on_uint16 = uint16(0)
        for i in range(15, -1, -1):
            bits_on_uint16 = bits_on_uint16 | (list_of_digits[i] << i)
        list_of_digits.clear()
        return bits_on_uint16

    @staticmethod
    def __flush_digit_list_for_frac_part(list_of_digits: [int]) -> uint16:
        bits_on_uint16 = uint16(0)
        for i in range(16):
            bits_on_uint16 = bits_on_uint16 | (list_of_digits[i] << (15 - i))
        list_of_digits.clear()
        return bits_on_uint16

    @staticmethod
    def __convert_integer_part_to_binary(int_part: int) -> tuple[int, list[uint16]]:
        bit_stream = []
        binary_digits_stack = [0] * 16
        offset = 0
        while int_part > 0:
            binary_digit = int_part % 2
            int_part //= 2
            binary_digits_stack[offset % 16] = binary_digit
            offset += 1

            if offset % 16 == 0 or int_part == 0:
                value = RealNumber.__flush_digit_list_for_int_part(binary_digits_stack)
                bit_stream.append(value)
                binary_digits_stack = [0] * 16

        bit_stream.reverse()
        return offset, bit_stream

    @staticmethod
    def __convert_fractional_part_to_binary(frac_part: int, available_mantissa_bit_size: int) -> tuple[
        int, list[uint16]]:
        bit_stream = []
        binary_digits_stack = [0] * 16
        offset = 0
        while frac_part > 0 and offset < available_mantissa_bit_size:
            pre = int(math.log10(frac_part))
            frac_part *= 2
            post = int(math.log10(frac_part))
            if post - pre:
                binary_digits_stack[offset % 16] = post - pre
                frac_part -= 10 ** post
            offset += 1
            if offset % 16 == 0 or frac_part == 0 or offset == available_mantissa_bit_size:
                value = RealNumber.__flush_digit_list_for_frac_part(binary_digits_stack)
                bit_stream.append(value)
                binary_digits_stack = [0] * 16

        return offset, bit_stream

    @staticmethod
    def __look_for_point(str_repr: str):
        length = len(str_repr)
        for idx in range(length):
            ch = str_repr[idx]
            if ch == '.':
                return idx
            elif not ('0' <= ch <= '9'):
                raise Exception("Not an decimal literal at string representation. ch = " + ch)
        return -1

    @staticmethod
    def __look_after_point(str_repr: str, idx: int):
        idx += 1
        length = len(str_repr)
        for i in range(idx, length):
            ch = str_repr[i]
            if not ('0' <= ch <= '9'):
                raise Exception("Not an decimal literal at string representation. ch = " + ch)

    @staticmethod
    def __get_parts_of_str_repr(str_repr: str):
        point_idx = RealNumber.__look_for_point(str_repr)
        integer_part: int

        if point_idx > 0:
            integer_part = int(str_repr[: point_idx])
        elif point_idx < 0:
            integer_part = int(str_repr)
        else:
            integer_part = 0

        fractional_part: int
        if point_idx >= 0:
            RealNumber.__look_after_point(str_repr, point_idx)
            fractional_part = int(str_repr[point_idx + 1:])
        else:
            fractional_part = 0

        return integer_part, fractional_part


