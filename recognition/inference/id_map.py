# -*- coding: utf-8 -*-
# 
# @Author   : Yuxiang WU
# @Email    : elephantameler@gmail.com


"""
    the key is label for a neuronal network, and the value is the abstract neural identity.
"""

# 140 + 1 ids
id_map = {0  : 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 20, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27, 25: 28,
          26 : 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 38, 35: 39, 36: 42, 37: 43, 38: 44, 39: 45, 40: 46, 41: 47, 42: 48, 43: 50, 44: 51, 45: 53, 46: 54, 47: 55, 48: 56, 49: 57,
          50 : 60, 51: 62, 52: 63, 53: 64, 54: 65, 55: 66, 56: 67, 57: 68, 58: 69, 59: 70, 60: 71, 61: 72, 62: 73, 63: 74, 64: 75, 65: 76, 66: 77, 67: 79, 68: 81, 69: 82, 70: 84, 71: 85, 72: 86, 73: 87,
          74 : 88, 75: 90, 76: 93, 77: 94, 78: 95, 79: 97, 80: 98, 81: 99, 82: 100, 83: 101, 84: 102, 85: 103, 86: 104, 87: 105, 88: 106, 89: 108, 90: 127, 91: 129, 92: 130, 93: 131, 94: 132, 95: 133,
          96 : 135, 97: 138, 98: 139, 99: 140, 100: 141, 101: 142, 102: 143, 103: 145, 104: 149, 105: 200, 106: 201, 107: 202, 108: 203, 109: 204, 110: 205, 111: 207, 112: 208, 113: 209, 114: 211,
          115: 212, 116: 213, 117: 214, 118: 216, 119: 225, 120: 229, 121: 233, 122: 234, 123: 236, 124: 975, 125: 976, 126: 977, 127: 978, 128: 979, 129: 980, 130: 981, 131: 983, 132: 989, 133: 990,
          134: 991, 135: 993, 136: 996, 137: 997, 138: 999, 139: 1000, 140: -1}

pi = list(id_map.values())[:-1]
