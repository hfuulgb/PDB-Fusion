import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


x=[264, 225, 430, 200, 129, 1058, 133, 662, 275, 309, 970, 162, 171, 331, 425, 163, 158, 267, 445, 179, 155, 234, 325, 378, 801, 69, 148, 104, 973, 262, 157, 118, 76, 950, 218, 320, 287, 341, 290, 270, 1016, 279, 1510, 303, 207, 681, 106, 108, 695, 173, 238, 151, 365, 124, 169, 154, 113, 1396, 374, 1550, 368, 904, 884, 330, 240, 530, 321, 115, 122, 168, 311, 263, 195, 464, 293, 145, 1793, 604, 477, 356, 251, 159, 622, 256, 215, 1227, 398, 236, 253, 102, 816, 298, 147, 814, 505, 438, 396, 268, 80, 808, 376, 332, 685, 534, 500, 206, 472, 338, 319, 214, 91, 778, 514, 377, 352, 161, 72, 1263, 891, 513, 484, 452, 391, 381, 73, 1627, 553, 488, 105, 942, 799, 729, 401, 388, 237, 736, 537, 383, 362, 310, 273, 196, 902, 872, 427, 302, 1225, 1177, 939, 498, 474, 419, 347, 892, 843, 716, 355, 260, 247, 164, 625, 521, 416, 392, 286, 230, 138, 2305, 1166, 1014, 879, 807, 493, 482, 460, 451, 436, 342, 328, 243, 137, 1107, 868, 844, 693, 628, 602, 453, 363, 334, 333, 323, 198, 165, 150, 116, 64, 805, 775, 765, 758, 435, 415, 406, 372, 364, 339, 337, 211, 156, 1002, 791, 771, 707, 678, 673, 637, 593, 543, 542, 541, 420, 349, 305, 280, 274, 219, 121, 1997, 1370, 1240, 1209, 1009, 955, 920, 875, 845, 636, 568, 532, 483, 414, 404, 393, 380, 307, 281, 232, 212, 189, 180, 141, 1970, 1333, 1198, 1158, 1151, 1140, 964, 708, 644, 626, 621, 586, 560, 531, 495, 489, 480, 475, 440, 426, 346, 283, 257, 255, 252, 242, 233, 224, 201, 90, 88, 933, 883, 873, 800, 743, 727, 713, 638, 619, 596, 554, 540, 465, 458, 447, 373, 360, 335, 313, 312, 296, 292, 248, 203, 186, 183, 178, 99, 57, 54, 1883, 1878, 1187, 1124, 961, 876, 839, 836, 803, 740, 665, 605, 507, 476, 468, 441, 434, 408, 384, 345, 318, 250, 228, 226, 193, 185, 166, 123, 119, 1586, 1235, 1173, 1129, 1073, 983, 930, 887, 828, 786, 714, 623, 615, 582, 561, 551, 525, 510, 479, 470, 462, 446, 437, 411, 402, 400, 394, 389, 371, 367, 366, 358, 357, 353, 350, 324, 291, 288, 284, 276, 272, 239, 197, 181, 128, 114, 2000, 1502, 1365, 1302, 1215, 1169, 1145, 1074, 1062, 1028, 1026, 994, 984, 952, 900, 829, 815, 759, 745, 700, 698, 655, 642, 634, 601, 589, 573, 529, 527, 526, 522, 516, 501, 490, 486, 449, 424, 410, 354, 351, 317, 282, 216, 210, 209, 125, 109, 95, 66, 1772, 1622, 1556, 1462, 1380, 1351, 1244, 1190, 1163, 1161, 1098, 1032, 1008, 991, 940, 919, 917, 898, 863, 862, 817, 796, 785, 770, 748, 680, 653, 646, 613, 585, 579, 577, 564, 552, 548, 519, 499, 478, 459, 455, 443, 442, 417, 412, 390, 387, 336, 329, 327, 314, 308, 300, 277, 266, 254, 246, 229, 227, 221, 190, 152, 146, 144, 131, 127, 63, 55, 3726, 1534, 1515, 1488, 1487, 1409, 1239, 1234, 1192, 1188, 1184, 1142, 1132, 1037, 988, 958, 911, 897, 885, 882, 859, 840, 832, 773, 772, 766, 757, 752, 715, 660, 658, 650, 648, 647, 641, 632, 618, 610, 599, 575, 567, 559, 528, 512, 487, 473, 454, 431, 428, 399, 370, 348, 326, 306, 285, 269, 261, 259, 249, 194, 174, 170, 143, 142, 93, 86, 81, 78, 71, 60, 3418, 2498, 2492, 1891, 1845, 1738, 1522, 1499, 1395, 1394, 1378, 1324, 1320, 1280, 1255, 1208, 1205, 1200, 1170, 1066, 1055, 998, 993, 928, 912, 848, 846, 811, 804, 764, 763, 762, 761, 737, 731, 719, 710, 663, 651, 649, 643, 631, 607, 584, 574, 571, 566, 562, 558, 550, 545, 539, 518, 506, 492, 485, 481, 471, 466, 433, 432, 423, 421, 413, 395, 359, 344, 343, 304, 301, 297, 295, 289, 278, 245, 244, 217, 205, 204, 191, 176, 175, 135, 134, 112, 107, 101, 100, 94, 92, 77, 75, 62, 61, 56, 3066, 3011, 2715, 2654, 2582, 2446, 2285, 2148, 1826, 1702, 1700, 1689, 1605, 1603, 1581, 1561, 1548, 1530, 1520, 1483, 1436, 1419, 1417, 1342, 1339, 1299, 1281, 1250, 1236, 1217, 1196, 1189, 1180, 1154, 1146, 1139, 1120, 1119, 1097, 1094, 1082, 1081, 1059, 1050, 1046, 1036, 1000, 997, 969, 963, 956, 953, 951, 946, 943, 941, 918, 915, 910, 903, 874, 869, 858, 834, 825, 806, 802, 798, 790, 787, 784, 782, 753, 735, 733, 726, 725, 722, 721, 718, 711, 705, 704, 701, 696, 692, 688, 677, 674, 670, 664, 652, 640, 630, 606, 600, 595, 594, 591, 580, 572, 570, 556, 533, 524, 520, 517, 509, 508, 469, 461, 450, 439, 429, 422, 405, 403, 382, 379, 375, 361, 340, 316, 315, 271, 265, 258, 241, 231, 222, 220, 213, 202, 199, 188, 184, 177, 172, 167, 160, 139, 126, 117, 110, 98, 96, 84, 82, 79, 65, 4911, 4377, 4144, 3969, 3828, 3664, 3130, 3026, 2897, 2723, 2718, 2514, 2511, 2444, 2266, 2187, 2180, 2136, 2048, 2002, 1987, 1978, 1972, 1905, 1803, 1766, 1765, 1761, 1756, 1719, 1718, 1668, 1596, 1590, 1575, 1559, 1547, 1513, 1475, 1468, 1467, 1455, 1439, 1412, 1410, 1402, 1392, 1360, 1318, 1314, 1310, 1285, 1276, 1274, 1267, 1254, 1246, 1245, 1243, 1242, 1237, 1223, 1203, 1199, 1191, 1178, 1171, 1165, 1162, 1155, 1148, 1137, 1135, 1130, 1111, 1105, 1104, 1093, 1088, 1077, 1075, 1070, 1067, 1061, 1048, 1043, 1035, 1031, 1018, 1015, 1012, 1007, 1006, 1003, 1001, 990, 982, 980, 979, 971, 954, 948, 922, 921, 914, 913, 909, 905, 895, 889, 881, 870, 864, 856, 1416, 847, 841, 833, 831, 1108, 822, 819, 813, 810, 809, 789, 781, 780, 754, 742, 741, 739, 734, 732, 724, 703, 699, 694, 691, 690, 689, 686, 683, 682, 676, 675, 668, 667, 666, 657, 656, 645, 624, 620, 614, 612, 609, 597, 590, 587, 583, 581, 578, 576, 569, 557, 555, 544, 538, 535, 523, 504, 503, 497, 496, 494, 491, 467, 457, 456, 448, 444, 418, 407, 397, 385, 322, 299, 294, 235, 223, 208, 153, 149, 140, 136, 132, 130, 120, 111, 103, 97, 89, 87, 85, 4662, 3925, 3856, 3574, 3473, 3230, 3198, 3159, 3072, 3005, 2947, 2716, 2669, 2649, 2572, 2531, 2517, 2510, 2506, 2452, 2408, 2395, 2380, 2346, 2321, 2289, 2248, 2245, 2223, 2193, 2176, 2175, 2168, 2150, 2130, 2129, 2082, 2055, 2045, 2033, 1957, 1953, 1914, 1910, 1907, 1895, 1890, 1867, 1863, 1862, 1849, 1835, 1832, 1830, 1829, 1798, 1787, 1778, 1773, 1755, 1752, 1740, 1537, 1724, 1723, 1713, 1711, 1703, 1698, 1687, 1671, 1638, 1634, 1629, 1624, 1623, 1613, 1612, 1608, 1601, 1598, 1589, 1579, 1538, 1523, 1518, 1512, 1509, 1507, 1506, 1504, 1493, 1480, 1479, 1469, 1459, 1456, 1451, 1450, 1444, 1440, 1433, 1428, 1415, 1405, 1404, 1401, 1397, 1393, 1387, 1384, 1383, 1377, 1373, 1372, 1371, 1369, 1368, 1364, 1363, 1359, 1358, 1347, 1341, 1337, 1332, 1330, 1328, 1327, 1323, 1312, 1311, 1308, 1305, 1301, 1297, 1295, 1293, 1292, 1288, 1287, 1284, 1282, 1273, 1272, 1271, 1264, 1260, 1259, 1257, 1230, 1228, 1226, 1214, 1212, 1211, 1206, 1202, 1201, 1197, 1195, 1186, 1185, 1183, 1181, 1174, 1168, 1167, 1164, 1160, 1156, 1152, 1147, 1143, 1141, 1134, 1133, 1131, 1125, 1123, 1122, 1121, 1118, 1117, 1109, 1106, 1102, 1101, 1100, 1099, 1096, 1090, 1085, 1084, 1080, 1072, 1071, 1068, 1063, 1060, 1056, 1054, 1053, 1047, 1045, 1044, 1042, 1033, 1030, 1029, 1027, 1025, 1024, 1022, 1021, 1019, 1017, 1013, 999, 996, 995, 992, 989, 986, 985, 978, 977, 976, 975, 972, 967, 962, 960, 959, 949, 947, 945, 944, 938, 936, 935, 934, 932, 931, 929, 926, 925, 924, 923, 916, 908, 906, 901, 894, 893, 890, 888, 886, 880, 878, 877, 871, 867, 865, 861, 860, 857, 855, 854, 853, 852, 851, 849, 842, 838, 830, 827, 826, 824, 821, 818, 792, 812, 797, 795, 794, 793, 788, 783, 777, 776, 769, 767, 760, 755, 751, 747, 746, 744, 738, 730, 728, 723, 720, 717, 712, 709, 706, 702, 697, 687, 684, 672, 671, 669, 661, 659, 654, 639, 635, 633, 629, 627, 617, 616, 611, 608, 603, 598, 592, 588, 565, 563, 549, 547, 546, 536, 515, 511, 502, 463, 409, 386, 369, 192, 187, 182, 83, 47, 74, 70, 67, 59, 58, 53, 52, 51, 774, 2763, 1343, 2113, 1856, 1931, 1379, 1159, 68, 1307, 1204, 1224, 2151, 1220, 1069, 1354, 2295, 1398, 2513, 907, 749, 1400, 1011, 2202, 1407, 1389, 2594, 1157, 835, 1413, 1064, 2190, 2068, 3092, 1076, 1172, 1218, 1585, 1486, 1386, 2169, 1262, 768, 1216, 1193, 750, 1353, 4095, 1532, 899, 1631, 1213, 1322, 1207, 2122, 1388, 1577, 1727, 1435, 1675, 1057, 2414, 1716, 2552, 1382, 1838, 1325, 1457, 2324, 1114, 1238, 1034, 1414, 937, 1855, 1194, 1391, 4116, 3177, 2551, 1473, 1010, 1986, 987, 1005, 1222, 1249, 1708, 1438, 1366, 779, 1256, 679, 1399, 3371, 1038, 1411, 1868, 1357, 1633, 1113, 3390, 1065, 1286, 4639, 1452, 1345, 1859, 1637, 1020, 2005, 1083, 1376, 3093, 1313, 1403, 1465, 2885, 1406, 1489, 1385, 2157, 1651, 837, 1894, 1610, 2027, 1424, 1650, 1657, 1621, 2127, 2156, 1466, 1375, 1913, 1500, 2147, 1454, 1023, 1555]

y=[22, 26, 20, 15, 31, 1, 33, 6, 29, 26, 6, 31, 18, 37, 15, 13, 28, 13, 15, 39, 38, 21, 26, 18, 4, 12, 31, 35, 1, 24, 24, 34, 20, 7, 26, 19, 20, 36, 24, 23, 6, 26, 1, 22, 36, 11, 28, 28, 3, 10, 31, 29, 32, 28, 21, 27, 33, 3, 28, 1, 24, 4, 3, 35, 21, 6, 27, 43, 54, 20, 27, 26, 30, 21, 27, 22, 1, 6, 13, 26, 29, 40, 9, 22, 26, 2, 27, 38, 20, 52, 4, 32, 49, 6, 22, 11, 28, 22, 22, 5, 29, 17, 2, 9, 12, 25, 15, 29, 16, 32, 20, 4, 20, 19, 26, 35, 15, 2, 5, 16, 9, 15, 25, 6, 11, 1, 11, 17, 20, 3, 4, 4, 18, 17, 34, 3, 8, 20, 31, 29, 34, 36, 5, 7, 20, 29, 2, 5, 5, 12, 16, 13, 19, 7, 7, 6, 22, 19, 25, 27, 7, 14, 21, 11, 28, 18, 32, 1, 3, 5, 3, 4, 17, 13, 8, 13, 17, 22, 22, 19, 24, 2, 4, 4, 5, 8, 11, 18, 22, 25, 27, 25, 28, 28, 28, 28, 18, 6, 5, 4, 6, 15, 16, 23, 23, 27, 24, 29, 32, 46, 4, 3, 4, 9, 5, 8, 12, 9, 13, 11, 15, 19, 18, 27, 25, 27, 26, 36, 1, 1, 1, 2, 4, 4, 5, 4, 5, 5, 13, 12, 5, 28, 22, 24, 25, 27, 20, 35, 22, 26, 32, 42, 2, 3, 3, 2, 3, 4, 5, 9, 9, 8, 4, 6, 12, 10, 13, 10, 8, 14, 12, 21, 29, 27, 27, 16, 34, 26, 24, 23, 28, 30, 18, 3, 6, 3, 4, 9, 6, 9, 8, 13, 8, 14, 8, 11, 15, 12, 19, 32, 28, 25, 34, 34, 31, 24, 16, 31, 23, 38, 21, 7, 5, 3, 2, 4, 3, 2, 4, 5, 10, 7, 5, 5, 9, 12, 22, 15, 11, 16, 15, 20, 22, 23, 25, 24, 32, 18, 72, 26, 40, 52, 1, 4, 1, 4, 3, 6, 2, 3, 5, 5, 11, 8, 5, 11, 8, 6, 9, 14, 12, 22, 8, 17, 14, 22, 16, 21, 25, 24, 26, 19, 28, 26, 21, 21, 26, 29, 29, 24, 20, 35, 27, 32, 22, 22, 22, 19, 1, 1, 4, 2, 2, 3, 1, 1, 2, 2, 3, 5, 2, 2, 3, 3, 5, 8, 3, 3, 8, 5, 7, 10, 10, 6, 9, 21, 16, 11, 9, 7, 10, 20, 20, 21, 10, 20, 27, 18, 21, 25, 36, 32, 24, 37, 16, 21, 24, 1, 1, 2, 1, 2, 2, 1, 2, 2, 3, 1, 2, 3, 1, 4, 5, 4, 4, 2, 7, 3, 6, 7, 7, 3, 6, 9, 5, 10, 8, 7, 3, 9, 12, 4, 10, 9, 14, 18, 13, 15, 11, 22, 18, 15, 22, 22, 28, 19, 22, 27, 27, 26, 10, 27, 30, 22, 34, 25, 35, 38, 28, 31, 25, 46, 18, 15, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 2, 2, 6, 1, 3, 5, 6, 4, 7, 6, 5, 2, 4, 3, 4, 3, 8, 4, 8, 11, 6, 5, 11, 10, 8, 9, 8, 7, 9, 13, 8, 10, 8, 11, 16, 19, 14, 16, 16, 26, 19, 22, 23, 21, 24, 25, 24, 20, 26, 35, 22, 20, 32, 48, 27, 15, 17, 30, 13, 15, 1, 1, 1, 1, 3, 2, 2, 3, 1, 1, 2, 2, 2, 2, 3, 2, 5, 1, 3, 3, 3, 2, 6, 6, 2, 2, 2, 6, 5, 4, 6, 9, 8, 5, 4, 8, 8, 11, 8, 6, 6, 9, 10, 8, 8, 10, 13, 10, 6, 12, 14, 8, 18, 12, 17, 17, 5, 11, 14, 18, 22, 17, 12, 20, 18, 32, 24, 20, 24, 29, 26, 29, 20, 25, 31, 36, 28, 26, 30, 28, 29, 24, 28, 31, 25, 26, 37, 25, 29, 37, 16, 14, 25, 11, 7, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 1, 3, 2, 4, 2, 2, 3, 4, 2, 1, 2, 2, 3, 2, 1, 4, 2, 3, 4, 3, 2, 5, 4, 3, 7, 3, 3, 4, 2, 5, 2, 1, 5, 4, 5, 3, 4, 5, 7, 4, 6, 2, 9, 11, 3, 6, 3, 5, 8, 7, 9, 6, 3, 5, 10, 9, 13, 9, 5, 8, 8, 5, 4, 6, 5, 7, 7, 7, 8, 10, 8, 11, 6, 11, 10, 6, 12, 14, 16, 12, 17, 13, 19, 16, 16, 19, 20, 24, 15, 20, 17, 31, 36, 28, 16, 27, 22, 22, 20, 23, 28, 25, 16, 11, 30, 29, 31, 27, 29, 25, 35, 26, 20, 27, 30, 43, 30, 27, 18, 14, 22, 15, 10, 22, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 1, 1, 1, 2, 3, 3, 2, 2, 1, 2, 2, 1, 1, 1, 1, 4, 1, 2, 1, 2, 2, 2, 1, 3, 2, 4, 1, 3, 2, 1, 4, 1, 2, 1, 5, 2, 3, 2, 4, 4, 4, 2, 3, 3, 6, 2, 2, 2, 3, 2, 4, 2, 2, 1, 1, 5, 5, 3, 5, 4, 2, 1, 2, 1, 2, 2, 6, 2, 4, 1, 2, 4, 1, 3, 1, 1, 3, 7, 4, 2, 3, 4, 2, 6, 4, 4, 3, 4, 6, 3, 6, 4, 7, 8, 4, 5, 4, 9, 9, 4, 6, 4, 4, 6, 10, 4, 3, 4, 9, 7, 6, 7, 7, 7, 2, 7, 10, 6, 7, 7, 5, 9, 10, 9, 4, 14, 10, 6, 10, 5, 4, 14, 10, 8, 12, 19, 12, 18, 24, 10, 10, 13, 13, 12, 22, 13, 26, 18, 20, 26, 28, 19, 20, 36, 26, 34, 25, 30, 25, 21, 42, 21, 15, 16, 18, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 6, 3, 2, 3, 1, 1, 1, 1, 4, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 3, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 3, 1, 2, 2, 2, 2, 4, 1, 1, 1, 1, 1, 1, 3, 3, 4, 2, 2, 3, 1, 1, 3, 1, 1, 2, 3, 8, 1, 1, 3, 2, 3, 1, 1, 1, 1, 1, 1, 2, 4, 2, 3, 1, 1, 4, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 1, 1, 1, 3, 3, 4, 1, 3, 1, 1, 1, 1, 4, 2, 2, 1, 1, 2, 2, 3, 3, 1, 1, 1, 1, 3, 2, 1, 2, 1, 1, 3, 2, 1, 3, 2, 1, 3, 1, 2, 2, 6, 2, 4, 1, 3, 2, 2, 3, 3, 8, 4, 2, 1, 3, 3, 4, 3, 4, 3, 2, 4, 3, 1, 1, 3, 7, 4, 2, 3, 4, 3, 5, 4, 5, 5, 2, 6, 2, 2, 4, 4, 2, 3, 3, 2, 2, 2, 5, 6, 4, 2, 2, 3, 2, 3, 2, 4, 4, 1, 3, 3, 5, 4, 3, 2, 1, 2, 3, 2, 8, 6, 6, 3, 8, 3, 5, 7, 7, 6, 7, 4, 7, 2, 5, 6, 3, 6, 6, 6, 5, 3, 5, 4, 3, 4, 2, 6, 6, 10, 5, 11, 10, 3, 12, 12, 7, 11, 10, 11, 14, 14, 22, 25, 30, 13, 1, 19, 13, 19, 9, 3, 3, 4, 12, 2, 1, 4, 1, 1, 1, 2, 1, 6, 1, 2, 3, 1, 2, 2, 1, 1, 1, 1, 1, 1, 6, 1, 1, 4, 2, 1, 2, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


sns.distplot(x)
plt.show()






