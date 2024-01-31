import pandas as pd
import numpy as np

def xyz_to_lab(xyz, xyz_white_ref=[95.047, 100.0, 108.883]):
    # 标准化
    xyz_normalized = xyz / xyz_white_ref

    # f(t)公式的辅助函数
    def f(t):
        delta = 6 / 29
        if t > delta ** 3:
            return t ** (1 / 3)
        else:
            return t / (3 * delta ** 2) + 4 / 29

    # 将 XYZ 转换为 Lab
    l = 116 * f(xyz_normalized[1]) - 16
    a = 500 * (f(xyz_normalized[0]) - f(xyz_normalized[1]))
    b = 200 * (f(xyz_normalized[1]) - f(xyz_normalized[2]))

    return np.array([l, a, b])

df_xyz = pd.read_csv(r'D:\pytorch learning\inverse design\1+5+1\dataset\dataset_xyz_1000k.csv')

lab_values = df_xyz.apply(lambda row: xyz_to_lab(row), axis=1)

df_lab = pd.DataFrame(lab_values.tolist(), columns=['L', 'a', 'b'])

df_lab.to_csv('input_CIE_lab_1000k.csv', index=False)
