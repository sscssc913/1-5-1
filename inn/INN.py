import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random

# 数据预处理
def load_and_process_data(file_path, scaler_thickness=None, encoder_materials1=None, encoder_materials2=None, scaler_outputs=None, fit_scaler=True):
    df = pd.read_csv(file_path, low_memory=False)

    for col in ['d2', 'd3', 'd4', 'd5', 'L', 'a', 'b']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    # 分割输入和输出
    inputs_thickness = df[['d2', 'd3', 'd4', 'd5']].values
    inputs_materials = df[['materials_3', 'materials_4']].copy()
    outputs = df[['L', 'a', 'b']].values

    # 归一化厚度
    if fit_scaler:
        scaler_thickness = MinMaxScaler()
        inputs_thickness = scaler_thickness.fit_transform(inputs_thickness)
    else:
        inputs_thickness = scaler_thickness.transform(inputs_thickness)

    # 对材料种类数据进行整数编码
    if fit_scaler:
        encoder_materials1 = LabelEncoder()
        encoder_materials2 = LabelEncoder()
        inputs_materials['materials_3'] = encoder_materials1.fit_transform(inputs_materials['materials_3'])
        inputs_materials['materials_4'] = encoder_materials2.fit_transform(inputs_materials['materials_4'])
    else:
        inputs_materials['materials_3'] = encoder_materials1.transform(inputs_materials['materials_3'])
        inputs_materials['materials_4'] = encoder_materials2.transform(inputs_materials['materials_4'])

    # 归一化输出
    if fit_scaler:
        scaler_outputs = MinMaxScaler()
        outputs = scaler_outputs.fit_transform(outputs)
    else:
        outputs = scaler_outputs.transform(outputs)

    return inputs_thickness, inputs_materials, outputs, scaler_thickness, encoder_materials1, encoder_materials2, scaler_outputs

# 多输入网络
class FNNWithEmbeddings(nn.Module):
    def __init__(self, num_materials, embedding_dim):
        super(FNNWithEmbeddings, self).__init__()
        self.material_embedding = nn.Embedding(num_embeddings=num_materials, embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim
        self.fc_thickness = nn.Linear(4, 128)

        self.fc1 = nn.Linear(128 + embedding_dim * 2, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, 500)
        self.fc7 = nn.Linear(500, 500)
        self.fc8 = nn.Linear(500, 3)

    def forward(self, x_thickness, x_material1, x_material2):
        material_embedding1 = self.material_embedding(x_material1)
        material_embedding2 = self.material_embedding(x_material2)
        x_thickness = torch.relu(self.fc_thickness(x_thickness))

        x = torch.cat([x_thickness, material_embedding1, material_embedding2], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x

# ΔE2000 计算公式
def delta_e_cie2000(lab1, lab2, kL=1, kC=1, kH=1):
    """
    计算两个 LAB 颜色值之间的 CIE ΔE 2000 色差。
    lab1, lab2: 两个颜色的 LAB 值。
    kL, kC, kH: 用于调整相对重要性的参数，默认为1。
    """
    # 将LAB值转换为弧度
    C1 = np.sqrt(lab1[1]**2 + lab1[2]**2)
    C2 = np.sqrt(lab2[1]**2 + lab2[2]**2)
    barC = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(barC**7 / (barC**7 + 25**7)))

    a1_prime = lab1[1] * (1 + G)
    a2_prime = lab2[1] * (1 + G)

    C1_prime = np.sqrt(a1_prime**2 + lab1[2]**2)
    C2_prime = np.sqrt(a2_prime**2 + lab2[2]**2)

    h1_prime = np.degrees(np.arctan2(lab1[2], a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(lab2[2], a2_prime)) % 360

    delta_L_prime = lab2[0] - lab1[0]
    delta_C_prime = C2_prime - C1_prime

    h_bar = abs(h1_prime - h2_prime)
    delta_h_prime = 0
    if C1_prime * C2_prime != 0:
        if h_bar <= 180:
            delta_h_prime = h2_prime - h1_prime
        elif h_bar > 180 and h2_prime <= h1_prime:
            delta_h_prime = h2_prime - h1_prime + 360
        else:
            delta_h_prime = h2_prime - h1_prime - 360
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime) / 2)

    L_bar_prime = (lab1[0] + lab2[0]) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2

    h_bar_prime = 0
    if C1_prime * C2_prime != 0:
        if h_bar <= 180:
            h_bar_prime = (h1_prime + h2_prime) / 2
        elif h_bar > 180 and (h1_prime + h2_prime) < 360:
            h_bar_prime = (h1_prime + h2_prime + 360) / 2
        else:
            h_bar_prime = (h1_prime + h2_prime - 360) / 2

    T = 1 - 0.17 * np.cos(np.radians(h_bar_prime - 30)) + \
        0.24 * np.cos(np.radians(2 * h_bar_prime)) + \
        0.32 * np.cos(np.radians(3 * h_bar_prime + 6)) - \
        0.20 * np.cos(np.radians(4 * h_bar_prime - 63))

    SL = 1 + ((0.015 * (L_bar_prime - 50)**2) /
              np.sqrt(20 + (L_bar_prime - 50)**2))
    SC = 1 + 0.045 * C_bar_prime
    SH = 1 + 0.015 * C_bar_prime * T

    RT = -2 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7)) * \
         np.sin(np.radians(60 * np.exp(-((h_bar_prime - 275) / 25)**2)))

    delta_E = np.sqrt(
        (delta_L_prime / (kL * SL))**2 +
        (delta_C_prime / (kC * SC))**2 +
        (delta_H_prime / (kH * SH))**2 +
        RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )
    return delta_E


def test_model(model, test_loader, scaler_outputs, device):
    model.eval()
    test_delta_es = []
    with torch.no_grad():
        for batch_thickness, batch_material1, batch_material2, batch_outputs in test_loader:
            batch_thickness, batch_material1, batch_material2, batch_outputs = \
                batch_thickness.to(device), batch_material1.to(device), batch_material2.to(device), batch_outputs.to(device)

            outputs_pred = model(batch_thickness, batch_material1, batch_material2)

            # 反归一化
            outputs_pred_np = scaler_outputs.inverse_transform(outputs_pred.detach().cpu().numpy())
            batch_outputs_np = scaler_outputs.inverse_transform(batch_outputs.cpu().numpy())

            # ΔE2000
            test_delta_es.extend([delta_e_cie2000(pred, exp) for pred, exp in zip(outputs_pred_np, batch_outputs_np)])
    return test_delta_es


class INN(nn.Module):
    def __init__(self, num_materials, embedding_dim, dropout_prob=0.5):
        super(INN, self).__init__()
        self.fc1 = nn.Linear(3, 250)
        self.bn1 = nn.BatchNorm1d(250)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(250, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(250, 250)
        self.bn3 = nn.BatchNorm1d(250)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.fc4 = nn.Linear(250, 250)
        self.bn4 = nn.BatchNorm1d(250)
        self.dropout4 = nn.Dropout(p=dropout_prob)
        self.fc_thickness = nn.Linear(250, 4)
        self.fc_materials = nn.Linear(250, num_materials * 2)

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(torch.relu(self.bn4(self.fc4(x))))
        thickness = self.fc_thickness(x)
        materials_logits = self.fc_materials(x)
        return thickness, materials_logits


# 随机种子
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def inn_loss_function(pred_thickness, true_thickness, pred_material_logits, true_material1, true_material2,
                      pred_lab, true_lab, epoch, criterion_mse, criterion_ce, num_materials):
    loss_thickness = criterion_mse(pred_thickness, true_thickness)

    pred_material_logits1 = pred_material_logits[:, :num_materials]
    pred_material_logits2 = pred_material_logits[:, num_materials:]

    # 材料的交叉熵损失
    loss_materials1 = criterion_ce(pred_material_logits1, true_material1)
    loss_materials2 = criterion_ce(pred_material_logits2, true_material2)

    # 损失相加
    loss_materials = (loss_materials1 + loss_materials2) / 2

    loss_lab = criterion_mse(pred_lab, true_lab)

    if epoch < 100:
        loss_total = loss_lab + 0.2 * loss_thickness + 0.1 * 0.5 *loss_materials
    else:
        loss_total = loss_lab
    return loss_total

def main():
    # 12种材料
    num_materials = 12
    embedding_dim = 8

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载和处理数据
    train_thickness, train_materials, train_outputs, scaler_thickness, encoder_materials1, encoder_materials2, scaler_outputs = load_and_process_data(
        r'D:\pytorch\INN\train_dataset_1000k.csv', fit_scaler=True)
    valid_thickness, valid_materials, valid_outputs, _, _, _, _ = load_and_process_data(
        r'D:\pytorch\INN\valid_dataset_100k.csv', scaler_thickness, encoder_materials1, encoder_materials2, scaler_outputs, fit_scaler=False)
    test_thickness, test_materials, test_outputs, _, _, _, _ = load_and_process_data(
        r'D:\pytorch\INN\test_dataset_10k.csv', scaler_thickness, encoder_materials1, encoder_materials2, scaler_outputs, fit_scaler=False)

    # 数据准备
    train_thickness_tensor = torch.tensor(train_thickness, dtype=torch.float32)
    train_materials_tensor = torch.tensor(train_materials.to_numpy(), dtype=torch.long)
    train_outputs_tensor = torch.tensor(train_outputs, dtype=torch.float32)
    valid_thickness_tensor = torch.tensor(valid_thickness, dtype=torch.float32)
    valid_materials_tensor = torch.tensor(valid_materials.to_numpy(), dtype=torch.long)
    valid_outputs_tensor = torch.tensor(valid_outputs, dtype=torch.float32)
    test_thickness_tensor = torch.tensor(test_thickness, dtype=torch.float32)
    test_materials_tensor = torch.tensor(test_materials.to_numpy(), dtype=torch.long)
    test_outputs_tensor = torch.tensor(test_outputs, dtype=torch.float32)

    # 数据加载器
    train_dataset = TensorDataset(train_thickness_tensor, train_materials_tensor[:, 0], train_materials_tensor[:, 1], train_outputs_tensor)
    valid_dataset = TensorDataset(valid_thickness_tensor, valid_materials_tensor[:, 0], valid_materials_tensor[:, 1], valid_outputs_tensor)
    test_dataset = TensorDataset(test_thickness_tensor, test_materials_tensor[:, 0], test_materials_tensor[:, 1], test_outputs_tensor)

    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    # 加载 FNN 模型
    fnn = FNNWithEmbeddings(num_materials=num_materials, embedding_dim=embedding_dim).to(device)
    fnn.load_state_dict(torch.load(r'D:\pytorch\INN\fnn_7_500_model.pth', map_location=device))
    fnn.eval()

    # cuDNN
    torch.backends.cudnn.benchmark = True

    # 存储损失和Delta E
    train_losses = []
    valid_losses = []
    valid_delta_es = []
    test_delta_es = []

    # 实例 INN
    inn = INN(num_materials=num_materials, embedding_dim=embedding_dim).to(device)
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    optimizer_inn = optim.Adam(inn.parameters(), lr=0.001)

    # 训练循环
    for epoch in tqdm(range(200), desc="Training Progress"):
        inn.train()
        epoch_train_losses = []
        for batch_thickness, batch_material1, batch_material2, batch_lab in train_loader:
            batch_thickness = batch_thickness.to(device)
            batch_material1 = batch_material1.to(device)
            batch_material2 = batch_material2.to(device)
            batch_lab = batch_lab.to(device)

            optimizer_inn.zero_grad()

            # INN
            pred_thickness, pred_material_logits = inn(batch_lab)

            pred_material1 = torch.argmax(pred_material_logits[:, :num_materials], dim=1)
            pred_material2 = torch.argmax(pred_material_logits[:, num_materials:], dim=1)

            # FNN
            pred_lab = fnn(pred_thickness, pred_material1, pred_material2)

            # 计算损失
            loss = inn_loss_function(pred_thickness, batch_thickness, pred_material_logits,
                                     batch_material1, batch_material2, pred_lab, batch_lab, epoch,
                                     criterion_mse, criterion_ce, num_materials)
            loss.backward()
            optimizer_inn.step()
            epoch_train_losses.append(loss.item())

        # 10个周期记录训练验证损失
        if (epoch + 1) % 10 == 0:
            train_loss_avg = np.mean(epoch_train_losses)
            train_losses.append(train_loss_avg)

            # 验证
            inn.eval()
            fnn.eval()
            with torch.no_grad():
                epoch_valid_losses = []
                for batch_thickness, batch_material1, batch_material2, batch_lab in valid_loader:
                    batch_thickness = batch_thickness.to(device)
                    batch_material1 = batch_material1.to(device)
                    batch_material2 = batch_material2.to(device)
                    batch_lab = batch_lab.to(device)

                    # INN
                    pred_thickness, pred_material_logits = inn(batch_lab)
                    pred_material1 = torch.argmax(pred_material_logits[:, :num_materials], dim=1)
                    pred_material2 = torch.argmax(pred_material_logits[:, num_materials:], dim=1)

                    # FNN
                    outputs_pred = fnn(batch_thickness, batch_material1, batch_material2)

                    # 计算损失
                    valid_loss = inn_loss_function(pred_thickness, batch_thickness, pred_material_logits,
                                                   batch_material1, batch_material2, outputs_pred, batch_lab, epoch,
                                                   criterion_mse, criterion_ce, num_materials)
                    epoch_valid_losses.append(valid_loss.item())

                valid_loss_avg = np.mean(epoch_valid_losses)
                valid_losses.append(valid_loss_avg)

                tqdm.write(
                    f'Epoch [{epoch + 1}/20], Training Loss: {train_loss_avg:.8f}, Validation Loss: {valid_loss_avg:.8f}')


        # 50个周期计算验证集上的平均delta_e
        if (epoch + 1) % 50 == 0:
            epoch_valid_delta_es = []
            for batch_thickness, batch_material1, batch_material2, batch_outputs in valid_loader:
                batch_thickness = batch_thickness.to(device)
                batch_material1 = batch_material1.to(device)
                batch_material2 = batch_material2.to(device)
                batch_outputs = batch_outputs.to(device)

                outputs_pred = fnn(batch_thickness, batch_material1, batch_material2)

                # 反归一化
                outputs_pred_np = scaler_outputs.inverse_transform(outputs_pred.detach().cpu().numpy())
                batch_outputs_np = scaler_outputs.inverse_transform(batch_outputs.cpu().numpy())

                # 计算 ΔE2000
                epoch_valid_delta_es.extend(
                    [delta_e_cie2000(pred, exp) for pred, exp in zip(outputs_pred_np, batch_outputs_np)])

            valid_delta_e_avg = np.mean(epoch_valid_delta_es)
            valid_delta_es.append(valid_delta_e_avg)
            tqdm.write(f'Epoch [{epoch + 1}/1000], Validation ΔE2000 Average: {valid_delta_e_avg:.8f}')

    # 测试，计算Delta E
    inn.eval()
    fnn.eval()
    test_delta_es = test_model(fnn, test_loader, scaler_outputs, device)
    test_delta_e_avg = np.mean(test_delta_es)
    tqdm.write(f'Test ΔE2000 Average: {test_delta_e_avg:.8f}')

    # 保存损失
    train_valid_loss_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Training_Loss': train_losses,
        'Validation_Loss': valid_losses,
    })
    train_valid_loss_df.to_csv('train_valid_loss_delta_e_7_500.csv', index=False)

    # 保存测试集上的Delta E和平均Delta E
    test_delta_e_df = pd.DataFrame({
        'Test_Delta_E': test_delta_es,
        'Test_Delta_E_Average': [test_delta_e_avg] * len(test_delta_es)
    })
    test_delta_e_df.to_csv('test_delta_e_results_7_500.csv', index=False)

    # 保存模型
    torch.save(fnn.state_dict(), 'fnn_7_500_model.pth')


if __name__ == '__main__':
    main()
