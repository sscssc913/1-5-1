import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simps
from multiprocessing import Pool

def load_csv(file_path):
    return pd.read_csv(file_path)

def calculate_xyz_for_range(args):
    df_range, cie_x_interp, cie_y_interp, cie_z_interp, source_psd_interp = args
    wavelengths = df_range.iloc[:, 0].to_numpy()
    transmittance = df_range.iloc[:, 1].to_numpy()

    X = simps(transmittance * source_psd_interp(wavelengths) * cie_x_interp(wavelengths), wavelengths)
    Y = simps(transmittance * source_psd_interp(wavelengths) * cie_y_interp(wavelengths), wavelengths)
    Z = simps(transmittance * source_psd_interp(wavelengths) * cie_z_interp(wavelengths), wavelengths)

    return {'X': X, 'Y': Y, 'Z': Z}

def main():
    df_transmittance = load_csv(r'D:\pytorch learning\inverse design\1+5+1\dataset\dataset_transmission_1000k.csv')
    df_cie_xyz = load_csv(r'D:\pytorch learning\inverse design\1+5+1\dataset\CIE_xyz_1931_2deg.csv')
    df_source_psd = load_csv(r'D:\pytorch learning\inverse design\1+5+1\dataset\CIE_std_illum_D65.csv')

    cie_x_interp = interp1d(df_cie_xyz.iloc[:, 0], df_cie_xyz.iloc[:, 1], kind='linear', bounds_error=False, fill_value=0)
    cie_y_interp = interp1d(df_cie_xyz.iloc[:, 0], df_cie_xyz.iloc[:, 2], kind='linear', bounds_error=False, fill_value=0)
    cie_z_interp = interp1d(df_cie_xyz.iloc[:, 0], df_cie_xyz.iloc[:, 3], kind='linear', bounds_error=False, fill_value=0)
    source_psd_interp = interp1d(df_source_psd.iloc[:, 0], df_source_psd.iloc[:, 1], kind='linear', bounds_error=False, fill_value=0)

    interval_size = (780 - 380) // 2 + 1  # 2nm一个数据点
    total_intervals = len(df_transmittance) // interval_size

    if len(df_transmittance) % interval_size != 0:
        total_intervals += 1

    # 分割数据集
    wavelength_ranges = [df_transmittance.iloc[i * interval_size: (i + 1) * interval_size] for i in range(total_intervals)]

    # 多进程计算
    tasks = [(df_range, cie_x_interp, cie_y_interp, cie_z_interp, source_psd_interp) for df_range in wavelength_ranges]

    with Pool() as pool:
        results = pool.map(calculate_xyz_for_range, tasks)

    xyz_df = pd.DataFrame(results)
    xyz_df.to_csv('dataset_xyz_1000k.csv', index=False)
    print("XYZ坐标已保存为 dataset_xyz_20000.csv")

if __name__ == '__main__':
    main()
