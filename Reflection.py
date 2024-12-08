import numpy as np
import multiprocessing
from functools import partial
import csv
from scipy.interpolate import interp1d
from materials import (get_SiO2_nk, get_Ti_nk, get_Al_nk)

# 计算反射率的函数
def compute_reflectance(l, SiO2_thickness_1, Ti_thickness_2, SiO2_thickness_3, Ti_thickness_4, SiO2_thickness_5):
    Al_thickness = 100e-9
    n0 = 1
    k0 = 2 * np.pi / l

    # 折射率和相位因子
    def calc_layer_params(thickness, n_func):
        n = n_func(l)
        phi = 2 * np.pi * n * thickness / l
        return n, phi

    n1, fai1 = calc_layer_params(SiO2_thickness_1, get_SiO2_nk)
    n2, fai2 = calc_layer_params(Ti_thickness_2, get_Ti_nk)
    n3, fai3 = calc_layer_params(SiO2_thickness_3, get_SiO2_nk)
    n4, fai4 = calc_layer_params(Ti_thickness_4, get_Ti_nk)
    n5, fai5 = calc_layer_params(SiO2_thickness_5, get_SiO2_nk)
    n6, fai6 = calc_layer_params(Al_thickness, get_Al_nk)
    ns = n1

    # 传播矩阵
    def transfer_matrix(n1, n2):
        return 0.5 * np.array([[1 + n2 / n1, 1 - n2 / n1], [1 - n2 / n1, 1 + n2 / n1]])

    # 相位矩阵
    P = lambda phi: np.array([[np.exp(-1j * phi), 0], [0, np.exp(1j * phi)]])

    # 总传播矩阵
    M_total = transfer_matrix(n0, n1) @ P(fai1) @ transfer_matrix(n1, n2) @ P(fai2) @ \
              transfer_matrix(n2, n3) @ P(fai3) @ transfer_matrix(n3, n4) @ P(fai4) @ \
              transfer_matrix(n4, n5) @ P(fai5) @ transfer_matrix(n5, n6) @ P(fai6) @ \
              transfer_matrix(n6, ns)

    # 反射率
    R = abs(M_total[1, 0] / M_total[0, 0]) ** 2

    l_nm = l * 1e9

    return l_nm, R


# 并行
def batched_parallel_compute(combinations, lambdas, batch_size=10000):
    max_processes = min(multiprocessing.cpu_count(), 60)
    pool = multiprocessing.Pool(max_processes)
    total_batches = len(combinations) // batch_size + (1 if len(combinations) % batch_size != 0 else 0)

    with open('Predicted_spectrum_TMM.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Lambda', 'Reflection'])

        for batch in range(total_batches):
            batch_combinations = combinations[batch * batch_size:(batch + 1) * batch_size]
            for combination in batch_combinations:
                func = partial(compute_reflectance, SiO2_thickness_1=combination[0], Ti_thickness_2=combination[1],
                               SiO2_thickness_3=combination[2], Ti_thickness_4=combination[3], SiO2_thickness_5=combination[4])
                batch_results = pool.map(func, lambdas)
                writer.writerows(batch_results)

    pool.close()
    pool.join()


def main():

    lambdas = np.linspace(400e-9, 1200e-9, 801)
    # 厚度
    combinations = [
        [136e-9, 7.2e-9, 143e-9, 16.3e-9, 144e-9],  # Blue
        [176e-9, 7.8e-9, 196e-9, 10.6e-9, 161e-9],  # Green
        [233e-9, 7.6e-9, 231e-9, 15.3e-9, 215e-9],  # Red
    ]

    batch_size = 10000
    batched_parallel_compute(combinations, lambdas, batch_size)


if __name__ == '__main__':
    main()
