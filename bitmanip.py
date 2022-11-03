import math

from numpy import uint8, uint64, dtype, uint16
import numpy as np


class BitMap:
    array = None
    elem_bit_width: int
    tot_bit_width: int
    arr_size: int

    def __init__(self, arr: np.array, py_list: list, py_int: int):

        if py_list:
            self.array = py_list
        elif py_int:
            self.array = BitMap.__to_np_array(py_int)
        else:
            self.array = arr

        self.arr_size = len(arr)
        self.elem_bit_width = arr[0].dtype.itemsize * 8
        self.tot_bit_width = self.elem_bit_width * self.arr_size

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise Exception("Index must be an integer.")
        if not 0 <= key < self.tot_bit_width:
            raise IndexError("Index %d is out of range from %d" % (key, self.tot_bit_width))

        element_contains_bit = key // self.elem_bit_width
        elem = self.array[element_contains_bit]

        if isinstance(elem, float):
            raise Exception("Numpy does not support bitwise shifting for floating point types.")

        elem_idx = key % self.elem_bit_width

        elem = int(elem)

        if value == 1:
            mask = 0x0001 << (self.elem_bit_width - 1)
            mask = mask >> elem_idx
            elem = elem | mask
        else:
            mask = 0x0000
            for i in range(self.elem_bit_width):
                if i == (self.elem_bit_width - 1 - elem_idx):
                    continue
                mask = mask | (1 << i)
            elem = elem & mask

        self.array[element_contains_bit] = elem

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise Exception("Index must be an integer.")
        if not 0 <= key < self.tot_bit_width:
            raise IndexError("Index %d is out of range from %d" % (key, self.tot_bit_width))

        element_contains_bit = key // self.elem_bit_width
        elem = self.array[element_contains_bit]

        if isinstance(elem, float):
            raise Exception("Numpy does not support bitwise shifting for floating point types.")

        elem_idx = key % self.elem_bit_width

        # See : https://github.com/numpy/numpy/issues/5668
        if isinstance(elem, uint64):
            # See : https://github.com/numpy/numpy/issues/10148
            elem = int(elem)
            while elem_idx < self.elem_bit_width - 1:
                elem //= 2
                elem_idx += 1
            return elem % 2

        """
        Reason for this weird behavior is numpy again.
        See : https://github.com/numpy/numpy/issues/21261
        Code can be as simple as:
            elem = elem << bit_idx    
            return elem >> self.__elem_bit_width - 1
        """
        elem = elem >> ((self.elem_bit_width - 1) - elem_idx)
        mask = uint8(0x0000)
        mask = uint8(mask | elem)
        mask = uint8(mask << 7)
        mask = uint8(mask >> 7)
        return mask

    @staticmethod
    def __to_np_array(py_int: int):
        bit_stream = np.zeros(int(math.log2(py_int)) // 16 + 1, dtype=uint16)
        for i in range(len(bit_stream) - 1, -1, -1):
            overflow = uint16(0x0000)
            overflow = overflow | bit_stream
            bit_stream >>= 16
            bit_stream[i] = overflow
        return bit_stream


def shift_bits_left(array, shift_amount: int):
    bit_width_of_array = array[0].dtype.itemsize * len(array)
    bitwise_copy(array, array, shift_amount, bit_width_of_array - shift_amount, 0)


def bitwise_copy(source, dest, offset: int, size: int, dest_offset: int):
    bitmap_src = BitMap(source)
    bitmap_dst = BitMap(dest)

    for i in range(offset, offset + size, 1):
        bitmap_dst[dest_offset + i] = bitmap_src[i]
