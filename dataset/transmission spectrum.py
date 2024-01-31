import numpy as np
import multiprocessing
from functools import partial
import csv
import random

from materials import (get_Ag_nk, get_SiO2_nk, get_Al_nk, get_Si_nk, get_Au_nk,
                       get_Cr_nk, get_Cu_nk, get_Ge_nk, get_Mo_nk, get_Ni_nk,
                       get_Pb_nk, get_Pt_nk, get_W_nk, get_ZnS_nk)

def get_random_material():
    materials = ["Ag", "Al", "Si", "Au", "Cr", "Cu", "Ge", "Mo", "Ni", "Pb", "Pt", "W"]
    return random.choice(materials)

material_to_function = {
    "Ag": get_Ag_nk, "Al": get_Al_nk, "Si": get_Si_nk, "Au": get_Au_nk,
    "Cr": get_Cr_nk, "Cu": get_Cu_nk, "Ge": get_Ge_nk, "Mo": get_Mo_nk,
    "Ni": get_Ni_nk, "Pb": get_Pb_nk, "Pt": get_Pt_nk, "W": get_W_nk,
    "ZnS": get_ZnS_nk, "SiO2": get_SiO2_nk
}

def compute_properties_for_combination(d_list, lambda_range):
    results = []
    random_material_1 = get_random_material()
    random_material_2 = get_random_material()

    material_functions = [
        get_ZnS_nk,
        material_to_function[random_material_1],
        material_to_function[random_material_2],
        get_ZnS_nk,
        get_Al_nk,
        get_SiO2_nk
    ]

    for l in lambda_range:
        k0 = 2 * np.pi / l
        M_total = np.eye(2)
        n_air = lambda l: 1
        n_ZnS = get_ZnS_nk(l)
        M1 = 0.5 * np.array(
            [[1 + n_ZnS / n_air(l), 1 - n_ZnS / n_air(l)], [1 - n_ZnS / n_air(l), 1 + n_ZnS / n_air(l)]])

        for i in range(len(d_list) + 1):
            n = material_functions[i](l)
            if i == 0:
                M_total = M_total @ M1
            else:
                n_prev = material_functions[i - 1](l)
                M = 0.5 * np.array([[1 + n / n_prev, 1 - n / n_prev], [1 - n / n_prev, 1 + n / n_prev]])
                M_total = M_total @ M

            if i < len(d_list):
                d = d_list[i]
                phi = n * k0 * d
                P = np.array([[np.exp(-1j * phi), 0], [0, np.exp(1j * phi)]])
                M_total = M_total @ P

        T = abs(1 / M_total[0, 0]) ** 2

        results.append([l, d_list[0], d_list[1], d_list[2], d_list[3], random_material_1, random_material_2, T])

    return results

def parallel_compute(combinations, lambdas):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    compute_func = partial(compute_properties_for_combination, lambda_range=lambdas)
    results = pool.map(compute_func, combinations)
    pool.close()
    pool.join()
    return results

def main():
    lambdas = np.linspace(380e-9, 780e-9, 201)  # 波长范围，1nm
    num_combinations = 1000000
    d_ranges = [
        [0, 200e-9], [0, 51e-9], [0, 50e-9], [0, 1000e-9]
    ]
    combinations = [np.random.uniform(low, high, num_combinations) for low, high in d_ranges]
    combinations = np.array(combinations).T

    results = parallel_compute(combinations, lambdas)

    with open('optical_properties_1000k.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for result in results:
            for sub_result in result:
                l_nm = sub_result[0] * 1e9
                d_list_nm = [d * 1e9 for d in sub_result[1:5]]
                writer.writerow([l_nm] + d_list_nm + [sub_result[5], sub_result[6], sub_result[7]])

if __name__ == '__main__':
    main()
