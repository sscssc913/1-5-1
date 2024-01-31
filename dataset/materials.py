import numpy as np
from scipy.interpolate import interp1d

Ag_data = np.array([
    [187.9, 1.07, 1.212],
    [191.6, 1.1, 1.232],
    [195.3, 1.12, 1.255],
    [199.3, 1.14, 1.277],
    [203.3, 1.15, 1.296],
    [207.3, 1.18, 1.312],
    [211.9, 1.2, 1.325],
    [216.4, 1.22, 1.336],
    [221.4, 1.25, 1.342],
    [226.2, 1.26, 1.344],
    [231.3, 1.28, 1.357],
    [237.1, 1.28, 1.367],
    [242.6, 1.3, 1.378],
    [249, 1.31, 1.389],
    [255.1, 1.33, 1.393],
    [261.6, 1.35, 1.387],
    [268.9, 1.38, 1.372],
    [276.1, 1.41, 1.331],
    [284.4, 1.41, 1.264],
    [292.4, 1.39, 1.161],
    [300.9, 1.34, 0.964],
    [310.7, 1.13, 0.616],
    [320.4, 0.81, 0.392],
    [331.5, 0.17, 0.829],
    [342.5, 0.14, 1.142],
    [354.2, 0.1, 1.419],
    [367.9, 0.07, 1.657],
    [381.5, 0.05, 1.864],
    [397.4, 0.05, 2.07],
    [413.3, 0.05, 2.275],
    [430.5, 0.04, 2.462],
    [450.9, 0.04, 2.657],
    [471.4, 0.05, 2.869],
    [495.9, 0.05, 3.093],
    [520.9, 0.05, 3.324],
    [548.6, 0.06, 3.586],
    [582.1, 0.05, 3.858],
    [616.8, 0.06, 4.152],
    [659.5, 0.05, 4.483],
    [704.5, 0.04, 4.838],
    [756, 0.03, 5.242],
    [821.1, 0.04, 5.727],
    [892, 0.04, 6.312],
    [984, 0.04, 6.992],
    [1088, 0.04, 7.795],
    [1216, 0.09, 8.828],
    [1393, 0.13, 10.1],
    [1610, 0.15, 11.85],
    [1937, 0.24, 14.08]
])

SiO2_data = np.array([
    [300, 1.4878, 0],
    [305, 1.4864, 0],
    [310, 1.4851, 0],
    [315, 1.4839, 0],
    [320, 1.4827, 0],
    [325, 1.4816, 0],
    [330, 1.4806, 0],
    [335, 1.4796, 0],
    [340, 1.4787, 0],
    [345, 1.4778, 0],
    [350, 1.4769, 0],
    [355, 1.4761, 0],
    [360, 1.4753, 0],
    [365, 1.4745, 0],
    [370, 1.4738, 0],
    [375, 1.4731, 0],
    [380, 1.4725, 0],
    [385, 1.4719, 0],
    [390, 1.4713, 0],
    [395, 1.4707, 0],
    [400, 1.4701, 0],
    [405, 1.4696, 0],
    [410, 1.4691, 0],
    [415, 1.4686, 0],
    [420, 1.4681, 0],
    [425, 1.4676, 0],
    [430, 1.4672, 0],
    [435, 1.4668, 0],
    [440, 1.4663, 0],
    [445, 1.466, 0],
    [450, 1.4656, 0],
    [455, 1.4652, 0],
    [460, 1.4648, 0],
    [465, 1.4645, 0],
    [470, 1.4641, 0],
    [475, 1.4638, 0],
    [480, 1.4635, 0],
    [485, 1.4632, 0],
    [490, 1.4629, 0],
    [495, 1.4626, 0],
    [500, 1.4623, 0],
    [505, 1.4621, 0],
    [510, 1.4618, 0],
    [515, 1.4615, 0],
    [520, 1.4613, 0],
    [525, 1.461, 0],
    [530, 1.4608, 0],
    [535, 1.4606, 0],
    [540, 1.4603, 0],
    [545, 1.4601, 0],
    [550, 1.4599, 0],
    [555, 1.4597, 0],
    [560, 1.4595, 0],
    [565, 1.4593, 0],
    [570, 1.4591, 0],
    [575, 1.4589, 0],
    [590, 1.4584, 0],
    [595, 1.4582, 0],
    [600, 1.458, 0],
    [605, 1.4579, 0],
    [610, 1.4577, 0],
    [615, 1.4576, 0],
    [620, 1.4574, 0],
    [625, 1.4572, 0],
    [630, 1.4571, 0],
    [635, 1.457, 0],
    [640, 1.4568, 0],
    [645, 1.4567, 0],
    [650, 1.4565, 0],
    [655, 1.4564, 0],
    [660, 1.4563, 0],
    [665, 1.4561, 0],
    [670, 1.456, 0],
    [675, 1.4559, 0],
    [680, 1.4558, 0],
    [685, 1.4556, 0],
    [690, 1.4555, 0],
    [695, 1.4554, 0],
    [700, 1.4553, 0],
    [710, 1.4551, 0],
    [720, 1.4549, 0],
    [730, 1.4546, 0],
    [740, 1.4544, 0],
    [750, 1.4542, 0],
    [760, 1.454, 0],
    [770, 1.4539, 0],
    [780, 1.4537, 0],
    [790, 1.4535, 0],
    [800, 1.4533, 0],
])

Al_data = np.array([
 [1.240e+02, 4.400e-02, 1.178e+00],
 [1.310e+02, 4.900e-02, 1.286e+00],
 [1.380e+02, 5.600e-02, 1.402e+00],
 [1.460e+02, 6.300e-02, 1.527e+00],
 [1.550e+02, 7.200e-02, 1.663e+00],
 [2.580e+02, 2.050e-01, 3.076e+00],
 [2.700e+02, 2.230e-01, 3.222e+00],
 [2.820e+02, 2.440e-01, 3.380e+00],
 [2.950e+02, 2.670e-01, 3.552e+00],
 [3.100e+02, 2.940e-01, 3.740e+00],
 [3.260e+02, 3.260e-01, 3.946e+00],
 [3.440e+02, 3.630e-01, 4.174e+00],
 [3.650e+02, 4.070e-01, 4.426e+00],
 [3.880e+02, 4.600e-01, 4.708e+00],
 [4.130e+02, 5.230e-01, 5.024e+00],
 [4.430e+02, 5.980e-01, 5.385e+00],
 [4.770e+02, 6.950e-01, 5.800e+00],
 [5.170e+02, 8.260e-01, 6.283e+00],
 [5.640e+02, 1.018e+00, 6.846e+00],
 [6.200e+02, 1.304e+00, 7.479e+00],
 [6.530e+02, 1.488e+00, 7.821e+00],
 [6.890e+02, 1.741e+00, 8.205e+00],
 [7.290e+02, 2.143e+00, 8.573e+00],
 [7.750e+02, 2.625e+00, 8.597e+00],
 [8.270e+02, 2.745e+00, 8.309e+00]])

Si_data = np.array([
    [300, 3.460073142, 3.340052421],
    [320, 3.947354703, 3.136057457],
    [340, 4.342046487, 2.831862719],
    [360, 4.623426929, 2.475487334],
    [380, 4.797423294, 2.111520679],
    [400, 4.884332737, 1.769541351],
    [420, 4.907842127, 1.463810996],
    [440, 4.887969162, 1.198366635],
    [460, 4.839824604, 0.971744486],
    [480, 4.774023996, 0.780256448],
    [500, 4.697842608, 0.619697809],
    [520, 4.616290205, 0.485938755],
    [540, 4.532817302, 0.375383865],
    [560, 4.449854944, 0.284770062],
    [580, 4.369123613, 0.211330607],
    [600, 4.291842979, 0.152602476],
    [620, 4.218858509, 0.106424605],
    [640, 4.150739123, 0.070908302],
    [660, 4.087816446, 0.044347754],
    [680, 4.030261995, 0.025276377],
    [700, 3.978072668, 0.012376062],
    [720, 3.931015842, 0.004514613],
    [740, 3.889155335, 0.00263341],
    [760, 3.855486982, 0.002668244],
    [780, 3.826652841, 0.002702097],
    [800, 3.801387912, 0.002734969]
])

Au_data = np.array([
    [288.3, 1.742, 1.9],
    [295.2, 1.776, 1.918],
    [302.4, 1.812, 1.92],
    [310, 1.83, 1.916],
    [317.9, 1.84, 1.904],
    [326.3, 1.824, 1.878],
    [335.1, 1.798, 1.86],
    [344.4, 1.766, 1.846],
    [354.2, 1.74, 1.848],
    [364.7, 1.716, 1.862],
    [375.7, 1.696, 1.906],
    [387.5, 1.674, 1.936],
    [400, 1.658, 1.956],
    [413.3, 1.636, 1.958],
    [427.5, 1.616, 1.94],
    [442.8, 1.562, 1.904],
    [459.2, 1.426, 1.846],
    [476.9, 1.242, 1.796],
    [495.9, 0.916, 1.84],
    [516.6, 0.608, 2.12],
    [539.1, 0.402, 2.54],
    [563.6, 0.306, 2.88],
    [590.4, 0.236, 2.97],
    [619.9, 0.194, 3.06],
    [652.6, 0.166, 3.15],
    [688.8, 0.16, 3.8],
    [729.3, 0.164, 4.35],
    [774.9, 0.174, 4.86],
    [826.6, 0.188, 5.39]
])

Cr_data = np.array([
    [295,   0.94,   2.58],
    [310,   1.02,   2.76],
    [326,   1.12,   2.95],
    [344,   1.26,   3.12],
    [365,   1.39,   3.24],
    [388,   1.44,   3.4],
    [413,   1.54,   3.71],
    [443,   1.8,   4.06],
    [477,   2.22,   4.36],
    [517,   2.75,   4.46],
    [564,   3.18,   4.41],
    [620,   3.48,   4.36],
    [701,   3.84,   4.37],
    [849,   4.31,   4.32]
])

Cu_data = np.array([
    [298.84201, 1.350727, 1.661416], [300.431732, 1.346555, 1.668262], [325.871063, 1.333079, 1.760546],
    [335.412537, 1.32612, 1.777938], [344.954865, 1.308167, 1.793567], [356.088501, 1.273463, 1.817923],
    [367.223083, 1.23066, 1.856502], [378.358459, 1.188992, 1.909461], [389.494598, 1.153467, 1.972805],
    [399.040344, 1.12988, 2.032445], [410.177551, 1.1118, 2.104793], [419.72403, 1.105274, 2.165856],
    [430.861908, 1.106555, 2.22948], [440.408844, 1.109853, 2.27483], [451.547058, 1.109618, 2.31884],
    [462.685303, 1.102885, 2.360813], [473.823486, 1.095451, 2.40459], [475.414642, 1.094729, 2.410908],
    [484.961487, 1.094306, 2.447223], [494.508057, 1.100383, 2.473184], [504.054382, 1.104697, 2.48236],
    [515.191345, 1.09497, 2.471842], [526.327759, 1.055081, 2.444926], [535.872681, 0.988267, 2.423289],
    [547.00769, 0.877987, 2.418465], [556.55127, 0.76544, 2.440671], [566.094116, 0.646936, 2.490745],
    [577.226501, 0.514649, 2.582264], [588.357788, 0.399107, 2.70026], [599.487671, 0.303734, 2.834858],
    [610.616272, 0.228599, 2.978474], [620.15387, 0.17921, 3.10441], [629.690308, 0.142343, 3.230063],
    [639.225464, 0.116595, 3.353002], [650.348267, 0.097415, 3.490248], [659.880615, 0.087346, 3.601937],
    [670.999878, 0.080279, 3.72554], [680.528992, 0.076543, 3.826307], [690.056519, 0.073758, 3.922985],
    [709.106689, 0.067786, 4.107314], [723.389648, 0.061674, 4.241879], [732.909302, 0.057572, 4.331265],
    [742.426941, 0.053981, 4.420626], [751.942688, 0.051242, 4.509973], [764.627014, 0.049361, 4.628936],
    [777.307495, 0.049889, 4.747389], [782.061646, 0.050769, 4.791586], [791.568237, 0.0537, 4.879454],
    [801.072388, 0.058234, 4.966377]
])

Ge_data = np.array([
    [250.0, 1.40845, 3.2529],
    [300.0, 3.84312, 3.7365],
    [350.0, 4.00377, 2.7056],
    [400.0, 4.14058, 2.2144],
    [450.0, 4.05033, 2.2057],
    [500.0, 4.388693, 2.4036],
    [550.0, 5.16247, 2.2107],
    [600.0, 5.72629, 1.3953],
    [650.0, 5.31663, 0.65482],
    [700.0, 5.01537, 0.46894],
    [750.0, 4.83073, 0.37173],
    [800.0, 4.70524, 0.32465]
])

Mo_data = np.array([
    [291.728, 0.5472, 3.4658],
    [309.96, 0.5009, 3.7324],
    [330.625, 0.4625, 4.0454],
    [354.241, 0.4353, 4.4099],
    [381.49, 0.4219, 4.8341],
    [413.281, 0.4253, 5.3305],
    [450.852, 0.4501, 5.9171],
    [495.937, 0.5043, 6.622],
    [551.041, 0.6032, 7.487],
    [619.921, 0.7797, 8.5785],
    [708.481, 1.1106, 10.0036],
    [826.561, 1.8114, 11.9264]
])

Ni_data = np.array([
    [294.943, 1.68, 1.87],
    [314.943, 1.67, 1.92],
    [324.943, 1.66, 1.97],
    [334.943, 1.65, 2.02],
    [344.943, 1.64, 2.07],
    [354.943, 1.63, 2.12],
    [365.382, 1.62, 2.17],
    [376.455, 1.61, 2.23],
    [388.219, 1.61, 2.3],
    [400.742, 1.61, 2.36],
    [414.1, 1.61, 2.44],
    [428.379, 1.62, 2.52],
    [443.679, 1.63, 2.61],
    [460.111, 1.64, 2.71],
    [477.808, 1.65, 2.81],
    [496.92, 1.67, 2.93],
    [517.625, 1.71, 3.06],
    [540.13, 1.75, 3.19],
    [564.682, 1.8, 3.33],
    [591.571, 1.85, 3.48],
    [621.15, 1.92, 3.65],
    [653.842, 2.02, 3.82],
    [690.167, 2.14, 4.01],
    [730.765, 2.28, 4.18],
    [776.437, 2.43, 4.31],
    [828.2, 2.53, 4.47]
])

Pb_data = np.array([
    [292.0, 1.18, 2.23],
    [301.0, 1.2, 2.29],
    [311.0, 1.21, 2.35],
    [320.0, 1.21, 2.42],
    [332.0, 1.2, 2.5],
    [342.0, 1.22, 2.57],
    [354.0, 1.23, 2.65],
    [368.0, 1.24, 2.74],
    [381.0, 1.26, 2.83],
    [397.0, 1.3, 2.93],
    [413.0, 1.33, 3.03],
    [431.0, 1.37, 3.14],
    [451.0, 1.41, 3.26],
    [471.0, 1.46, 3.39],
    [496.0, 1.52, 3.54],
    [521.0, 1.57, 3.68],
    [549.0, 1.64, 3.84],
    [582.0, 1.68, 4.02],
    [617.0, 1.75, 4.21],
    [659.0, 1.8, 4.42],
    [704.0, 1.86, 4.65],
    [756.0, 1.95, 4.89],
    [821.0, 2.06, 5.19]
])

Pt_data = np.array([
    [309.96, 1.1505, 2.8379],
    [330.625, 1.304, 3.1046],
    [354.241, 1.5367, 3.0624],
    [381.49, 1.2882, 2.908],
    [413.281, 0.8675, 3.2129],
    [450.852, 0.6273, 3.7605],
    [495.937, 0.5124, 4.3965],
    [551.041, 0.4643, 5.121],
    [619.921, 0.4611, 5.9757],
    [708.481, 0.5013, 7.0273],
    [826.561, 0.5979, 8.3824]
])

Ti_data = np.array([
    [300.0, 1.04125, 1.4996],
    [350.0, 1.27429, 1.9891],
    [400.0, 1.55031, 2.1501],
    [450.0, 1.69389, 2.2672],
    [500.0, 1.78553, 2.4051],
    [550.0, 1.88519, 2.6098],
    [600.0, 2.0425, 2.8062],
    [650.0, 2.21139, 2.9817],
    [700.0, 2.40904, 3.1459],
    [750.0, 2.63015, 3.2686],
    [800.0, 2.86107, 3.3169]
])

W_data = np.array([
    [299.5, 3.27, 2.21],
    [327.0, 3.17, 2.33],
    [347.5, 3.35, 2.28],
    [367.0, 3.28, 2.37],
    [387.5, 3.43, 2.35],
    [400.0, 3.39, 2.41],
    [413.3, 3.35, 2.42],
    [427.5, 3.32, 2.45],
    [442.8, 3.3, 2.49],
    [459.2, 3.31, 2.55],
    [476.9, 3.34, 2.62],
    [495.9, 3.38, 2.68],
    [516.6, 3.45, 2.72],
    [539.1, 3.5, 2.72],
    [563.6, 3.49, 2.75],
    [590.4, 3.54, 2.84],
    [619.9, 3.6, 2.89],
    [652.6, 3.7, 2.94],
    [688.8, 3.82, 2.91],
    [729.3, 3.84, 2.78],
    [774.9, 3.67, 2.68],
    [804.9, 3.59, 2.58]
])

ZnS_data = np.array([
    [298.84201, 2.783251, 0.334149],
    [300.431732, 2.778095, 0.328796],
    [321.100677, 2.744072, 0.267524],
    [340.183594, 2.757029, 0.198892],
    [360.860352, 2.750716, 0.096432],
    [381.540161, 2.683689, 0.02568],
    [400.631348, 2.608668, 0.006628],
    [421.315125, 2.545618, 0.008461],
    [440.408844, 2.503632, 0.00915],
    [461.094147, 2.469269, 0.009563],
    [480.18808, 2.444414, 0.010406],
    [500.872314, 2.422883, 0.011478],
    [521.555054, 2.405203, 0.01215],
    [540.644897, 2.391191, 0.012109],
    [561.322815, 2.378086, 0.011672],
    [580.406982, 2.367728, 0.011264],
    [601.077576, 2.358072, 0.010981],
    [620.15387, 2.350219, 0.01084],
    [640.814575, 2.342533, 0.010785],
    [661.469177, 2.335553, 0.010767],
    [680.528992, 2.329749, 0.010699],
    [701.169983, 2.324155, 0.010507],
    [720.216003, 2.319611, 0.01019],
    [740.84082, 2.315225, 0.009781],
    [761.456238, 2.311229, 0.009426],
    [780.476929, 2.30777, 0.009227],
    [801.072388, 2.304176, 0.009209]
])

# 提取波长、折射率和消光系数
wavelengths_Ag = Ag_data[:, 0] * 1e-9
n_values_Ag = Ag_data[:, 1]
k_values_Ag = Ag_data[:, 2]

wavelengths_SiO2 = SiO2_data[:, 0] * 1e-9
n_values_SiO2 = SiO2_data[:, 1]
k_values_SiO2 = SiO2_data[:, 2]

wavelengths_Al = Al_data[:, 0] * 1e-9
n_values_Al = Al_data[:, 1]
k_values_Al = Al_data[:, 2]

wavelengths_Si = Si_data[:, 0] * 1e-9
n_values_Si = Si_data[:, 1]
k_values_Si = Si_data[:, 2]

wavelengths_Au = Au_data[:, 0] * 1e-9
n_values_Au = Au_data[:, 1]
k_values_Au = Au_data[:, 2]

wavelengths_Cr = Cr_data[:, 0] * 1e-9
n_values_Cr = Cr_data[:, 1]
k_values_Cr = Cr_data[:, 2]

wavelengths_Cu = Cu_data[:, 0] * 1e-9
n_values_Cu = Cu_data[:, 1]
k_values_Cu = Cu_data[:, 2]

wavelengths_Ge = Ge_data[:, 0] * 1e-9
n_values_Ge = Ge_data[:, 1]
k_values_Ge = Ge_data[:, 2]

wavelengths_Mo = Mo_data[:, 0] * 1e-9
n_values_Mo = Mo_data[:, 1]
k_values_Mo = Mo_data[:, 2]

wavelengths_Ni = Ni_data[:, 0] * 1e-9
n_values_Ni = Ni_data[:, 1]
k_values_Ni = Ni_data[:, 2]

wavelengths_Pb = Pb_data[:, 0] * 1e-9
n_values_Pb = Pb_data[:, 1]
k_values_Pb = Pb_data[:, 2]

wavelengths_Pt = Pt_data[:, 0] * 1e-9
n_values_Pt = Pt_data[:, 1]
k_values_Pt = Pt_data[:, 2]

wavelengths_Ti = Ti_data[:, 0] * 1e-9
n_values_Ti = Ti_data[:, 1]
k_values_Ti = Ti_data[:, 2]

wavelengths_W = W_data[:, 0] * 1e-9
n_values_W = W_data[:, 1]
k_values_W = W_data[:, 2]

wavelengths_ZnS = ZnS_data[:, 0] * 1e-9
n_values_ZnS = ZnS_data[:, 1]
k_values_ZnS = ZnS_data[:, 2]

# 创建插值函数
n_interp_Ag = interp1d(wavelengths_Ag, n_values_Ag, kind='linear', fill_value="extrapolate")
k_interp_Ag = interp1d(wavelengths_Ag, k_values_Ag, kind='linear', fill_value="extrapolate")
n_interp_SiO2 = interp1d(wavelengths_SiO2, n_values_SiO2, kind='linear', fill_value="extrapolate")
k_interp_SiO2 = interp1d(wavelengths_SiO2, k_values_SiO2, kind='linear', fill_value="extrapolate")
n_interp_Al = interp1d(wavelengths_Al, n_values_Al, kind='linear', fill_value="extrapolate")
k_interp_Al = interp1d(wavelengths_Al, k_values_Al, kind='linear', fill_value="extrapolate")
n_interp_Si = interp1d(wavelengths_Si, n_values_Si, kind='linear', fill_value="extrapolate")
k_interp_Si = interp1d(wavelengths_Si, k_values_Si, kind='linear', fill_value="extrapolate")
n_interp_Au = interp1d(wavelengths_Au, n_values_Au, kind='linear', fill_value="extrapolate")
k_interp_Au = interp1d(wavelengths_Au, k_values_Au, kind='linear', fill_value="extrapolate")
n_interp_Cr = interp1d(wavelengths_Cr, n_values_Cr, kind='linear', fill_value="extrapolate")
k_interp_Cr = interp1d(wavelengths_Cr, k_values_Cr, kind='linear', fill_value="extrapolate")
n_interp_Cu = interp1d(wavelengths_Cu, n_values_Cu, kind='linear', fill_value="extrapolate")
k_interp_Cu = interp1d(wavelengths_Cu, k_values_Cu, kind='linear', fill_value="extrapolate")
n_interp_Ge = interp1d(wavelengths_Ge, n_values_Ge, kind='linear', fill_value="extrapolate")
k_interp_Ge = interp1d(wavelengths_Ge, k_values_Ge, kind='linear', fill_value="extrapolate")
n_interp_Mo = interp1d(wavelengths_Mo, n_values_Mo, kind='linear', fill_value="extrapolate")
k_interp_Mo = interp1d(wavelengths_Mo, k_values_Mo, kind='linear', fill_value="extrapolate")
n_interp_Ni = interp1d(wavelengths_Ni, n_values_Ni, kind='linear', fill_value="extrapolate")
k_interp_Ni = interp1d(wavelengths_Ni, k_values_Ni, kind='linear', fill_value="extrapolate")
n_interp_Pb = interp1d(wavelengths_Pb, n_values_Pb, kind='linear', fill_value="extrapolate")
k_interp_Pb = interp1d(wavelengths_Pb, k_values_Pb, kind='linear', fill_value="extrapolate")
n_interp_Pt = interp1d(wavelengths_Pt, n_values_Pt, kind='linear', fill_value="extrapolate")
k_interp_Pt = interp1d(wavelengths_Pt, k_values_Pt, kind='linear', fill_value="extrapolate")
n_interp_Ti = interp1d(wavelengths_Ti, n_values_Ti, kind='linear', fill_value="extrapolate")
k_interp_Ti = interp1d(wavelengths_Ti, k_values_Ti, kind='linear', fill_value="extrapolate")
n_interp_W = interp1d(wavelengths_W, n_values_W, kind='linear', fill_value="extrapolate")
k_interp_W = interp1d(wavelengths_W, k_values_W, kind='linear', fill_value="extrapolate")
n_interp_ZnS = interp1d(wavelengths_ZnS, n_values_ZnS, kind='linear', fill_value="extrapolate")
k_interp_ZnS = interp1d(wavelengths_ZnS, k_values_ZnS, kind='linear', fill_value="extrapolate")


def get_Ag_nk(lambda_val):
    n = n_interp_Ag(lambda_val)
    k = k_interp_Ag(lambda_val)
    return n + 1j * k

def get_SiO2_nk(lambda_val):
    n = n_interp_SiO2(lambda_val)
    return n

def get_Al_nk(lambda_val):
    n = n_interp_Al(lambda_val)
    k = k_interp_Al(lambda_val)
    return n + 1j * k

def get_Si_nk(lambda_val):
    n = n_interp_Si(lambda_val)
    k = k_interp_Si(lambda_val)
    return n + 1j * k

def get_Au_nk(lambda_val):
    n = n_interp_Au(lambda_val)
    k = k_interp_Au(lambda_val)
    return n + 1j * k

def get_Cr_nk(lambda_val):
    n = n_interp_Cr(lambda_val)
    k = k_interp_Cr(lambda_val)
    return n + 1j * k

def get_Cu_nk(lambda_val):
    n = n_interp_Cu(lambda_val)
    k = k_interp_Cu(lambda_val)
    return n + 1j * k

def get_Ge_nk(lambda_val):
    n = n_interp_Ge(lambda_val)
    k = k_interp_Ge(lambda_val)
    return n + 1j * k

def get_Mo_nk(lambda_val):
    n = n_interp_Mo(lambda_val)
    k = k_interp_Mo(lambda_val)
    return n + 1j * k

def get_Ni_nk(lambda_val):
    n = n_interp_Ni(lambda_val)
    k = k_interp_Ni(lambda_val)
    return n + 1j * k

def get_Pb_nk(lambda_val):
    n = n_interp_Pb(lambda_val)
    k = k_interp_Pb(lambda_val)
    return n + 1j * k

def get_Pt_nk(lambda_val):
    n = n_interp_Pt(lambda_val)
    k = k_interp_Pt(lambda_val)
    return n + 1j * k

def get_W_nk(lambda_val):
    n = n_interp_W(lambda_val)
    k = k_interp_W(lambda_val)
    return n + 1j * k

def get_ZnS_nk(lambda_val):
    n = n_interp_ZnS(lambda_val)
    k = k_interp_ZnS(lambda_val)
    return n + 1j * k