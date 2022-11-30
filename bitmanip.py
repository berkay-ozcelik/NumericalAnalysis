import math
import time

from numpy import uint8, uint64, dtype, uint16
import numpy as np


class BitMap:
    array = None
    elem_bit_width: int
    tot_bit_width: int
    arr_size: int

    def __init__(self, element):

        if isinstance(element, list):
            self.array = element
        elif isinstance(element, int):
            self.array = BitMap.__to_np_array(element)
        else:
            self.array = element

        self.arr_size = len(self.array)
        self.elem_bit_width = self.array[0].dtype.itemsize * 8
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
    def first_index_of(self,key):
        for i in range(len(self)):
            if self[i] == key:
                return i
        return -1
    def byte_size(self):
        return self.tot_bit_width // 8

    def get_byte(self, i: int) -> uint8:
        byte = [uint8(0x00)]
        byte_bitmap = BitMap(byte)
        bitwise_copy(self, byte_bitmap, i * 8, 8, 0)
        return byte[0]

    def set_byte(self, i: int, value: uint8):
        byte = [value]
        byte_bitmap = BitMap(byte)
        bitwise_copy(byte_bitmap, self, 0, 8, i * 8)

    def __str__(self):
        str_val = str(self[0])
        for i in range(1, self.tot_bit_width):
            if i % 8 == 0:
                str_val += ' ' + str(self[i])
            else:
                str_val += str(self[i])
        return str_val

    def __lshift__(self, sha: int):
        if sha > 8:
            n = sha // 8
            for i in range(n):
                self << 8
            sha %= 8

        mask = uint8(0x00)
        for i in range(sha):
            mask = mask | uint8(0x80) >> i

        prev_overflow = uint8(0x00)
        for i in range(self.byte_size() - 1, -1, -1):
            current = self.get_byte(i)
            bits_where_going_to_overflow = current & mask
            current = current << sha
            prev_overflow = prev_overflow >> (8 - sha)
            self.set_byte(i, uint8(current | prev_overflow))
            prev_overflow = bits_where_going_to_overflow
        return prev_overflow

    def __len__(self):
        return self.tot_bit_width

    @staticmethod
    def __to_np_array(py_int: int):
        if py_int < 0:
            raise ValueError("Can not convert negative integer to bitmap.")

        if py_int == 0:
            size = 1
        else:
            size = int(math.log2(py_int)) // 8 + 1

        bit_stream = np.zeros(size, dtype=uint8)

        shift_amount = 2 ** 8

        for i in range(len(bit_stream) - 1, -1, -1):
            overflow = uint8(0x00)
            tmp = py_int % shift_amount
            overflow = overflow | tmp
            py_int //= shift_amount
            bit_stream[i] = overflow
        return bit_stream


def bitwise_copy(src: BitMap, dest: BitMap, offset: int, size: int, dest_offset: int):
    j = 0
    for i in range(offset, offset + size, 1):
        dest[dest_offset + j] = src[i]
        j += 1


