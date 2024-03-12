import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# 数据预处理
def load_and_process_data(file_path, scaler_thickness=None, encoder_materials1=None, encoder_materials2=None, scaler_lab=None, fit_scaler=True):
    df = pd.read_csv(file_path, low_memory=False)

    for col in ['d2', 'd3', 'd4', 'd5', 'L', 'a', 'b']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    batch_thickness = df[['d2', 'd3', 'd4', 'd5']].values
    batch_materials1 = df['materials_3'].copy()
    batch_materials2 = df['materials_4'].copy()
    batch_lab = df[['L', 'a', 'b']].values

    # 归一化厚度
    if fit_scaler or scaler_thickness is None:
        scaler_thickness = MinMaxScaler()
        batch_thickness = scaler_thickness.fit_transform(batch_thickness)
    else:
        batch_thickness = scaler_thickness.transform(batch_thickness)

    # 材料种类进行整数编码
    if fit_scaler or encoder_materials1 is None or encoder_materials2 is None:
        encoder_materials1 = LabelEncoder()
        encoder_materials2 = LabelEncoder()
        batch_materials1_encoded = encoder_materials1.fit_transform(batch_materials1)
        batch_materials2_encoded = encoder_materials2.fit_transform(batch_materials2)
    else:
        batch_materials1_encoded = encoder_materials1.transform(batch_materials1)
        batch_materials2_encoded = encoder_materials2.transform(batch_materials2)

    # 归一化lab
    if fit_scaler or scaler_lab is None:
        scaler_lab = MinMaxScaler()
        batch_lab = scaler_lab.fit_transform(batch_lab)
    else:
        batch_lab = scaler_lab.transform(batch_lab)

    return batch_thickness, batch_lab, scaler_thickness, \
           encoder_materials1, encoder_materials2, scaler_lab, batch_materials1_encoded, batch_materials2_encoded


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


class INN(nn.Module):
    def __init__(self, num_materials, ):
        super(INN, self).__init__()
        self.fc1 = nn.Linear(3, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 250)
        self.fc4 = nn.Linear(250, 250)
        self.fc5 = nn.Linear(250, 250)
        self.fc6 = nn.Linear(250, 250)
        self.fc7 = nn.Linear(250, 250)
        self.fc_thickness = nn.Linear(250, 4)
        self.fc_materials = nn.Linear(250, num_materials * 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        pred_thickness = self.fc_thickness(x)
        pred_materials_logits = self.fc_materials(x)
        return pred_thickness, pred_materials_logits



def inn_loss_function(pred_thickness, true_thickness, pred_material_logits, true_materials1, true_materials2,
                      pred_lab, true_lab, criterion_mse, criterion_ce, num_materials, weight_thickness, weight_materials, weight_lab):
    loss_thickness = criterion_mse(pred_thickness, true_thickness)

    pred_material_logits1 = pred_material_logits[:, :num_materials]
    pred_material_logits2 = pred_material_logits[:, num_materials:]

    loss_materials1 = criterion_ce(pred_material_logits1, true_materials1)
    loss_materials2 = criterion_ce(pred_material_logits2, true_materials2)

    loss_materials = (loss_materials1 + loss_materials2) / 2

    loss_lab = criterion_mse(pred_lab, true_lab)

    loss_thickness = loss_thickness * weight_thickness
    loss_materials = loss_materials * weight_materials
    loss_lab = loss_lab * weight_lab

    total_loss = loss_thickness + loss_materials + loss_lab
    return total_loss, loss_thickness, loss_materials, loss_lab


def main():

    num_materials = 12
    embedding_dim = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_thickness, train_lab, scaler_thickness, \
    encoder_materials1, encoder_materials2, scaler_lab, \
    train_materials1_encoded, train_materials2_encoded = \
        load_and_process_data(r'D:\pytorch\INN\train_dataset_1000k.csv', fit_scaler=True)

    train_thickness_tensor = torch.tensor(train_thickness, dtype=torch.float32)
    train_materials1_tensor = torch.tensor(train_materials1_encoded, dtype=torch.long)
    train_materials2_tensor = torch.tensor(train_materials2_encoded, dtype=torch.long)
    train_lab_tensor = torch.tensor(train_lab, dtype=torch.float32)

    train_dataset = TensorDataset(train_thickness_tensor, train_materials1_tensor, train_materials2_tensor,
                                  train_lab_tensor)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True)

    fnn = FNNWithEmbeddings(num_materials=num_materials, embedding_dim=embedding_dim).to(device)
    fnn.load_state_dict(torch.load(r'D:\pytorch\INN\fnn_7_500_model.pth', map_location=device))
    fnn.eval()

    torch.backends.cudnn.benchmark = True

    inn = INN(num_materials=num_materials).to(device)
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    optimizer_inn = optim.Adam(inn.parameters(), lr=0.001)

    record_losses = {
        'Epoch': [],
        'Training_Loss': [],
        'Thickness_Loss': [],
        'Materials_Loss': [],
        'Lab_Loss': [],
    }

    # 训练循环
    epochs = 1000
    materials_epochs = 300
    thickness_epochs = 200


    for epoch in tqdm(range(1, epochs + 1), desc="Training Progress", total=epochs):

        if epoch <= materials_epochs:
            weight_thickness = 0
            weight_materials = 1
            weight_lab = 0
            calculate_accuracy = True
        elif epoch <= thickness_epochs + materials_epochs:
            weight_thickness = 0.2
            weight_materials = 0
            weight_lab = 1
            calculate_accuracy = False
        else:
            weight_thickness = 0
            weight_materials = 0
            weight_lab = 1.0
            calculate_accuracy = False

        epoch_train_losses = []
        epoch_accuracy = []
        epoch_losses_thickness = []
        epoch_losses_materials = []
        epoch_losses_lab = []

        for batch in train_loader:
            batch_thickness, batch_materials1_encoded, batch_materials2_encoded, batch_lab = [b.to(device) for b in batch]

            optimizer_inn.zero_grad()

            # 前向传播
            pred_thickness, pred_material_logits = inn(batch_lab)

            pred_materials1_logits = pred_material_logits[:, :12]
            pred_materials2_logits = pred_material_logits[:, 12:]

            pred_material1_label = torch.argmax(pred_materials1_logits, dim=1)
            pred_material2_label = torch.argmax(pred_materials2_logits, dim=1)
            pred_lab = fnn(pred_thickness, pred_material1_label, pred_material2_label)

            correct_predictions1 = (pred_material1_label == batch_materials1_encoded).float()
            accuracy1 = correct_predictions1.mean()
            correct_predictions2 = (pred_material2_label == batch_materials2_encoded).float()
            accuracy2 = correct_predictions2.mean()
            average_accuracy = (accuracy1 + accuracy2) / 2

            # 计算损失
            loss, loss_thickness, loss_materials, loss_lab = inn_loss_function(
                pred_thickness=pred_thickness,
                true_thickness=batch_thickness,
                pred_material_logits=pred_material_logits,
                true_materials1=batch_materials1_encoded,
                true_materials2=batch_materials2_encoded,
                pred_lab=pred_lab,
                true_lab=batch_lab,
                criterion_mse=criterion_mse,
                criterion_ce=criterion_ce,
                num_materials=num_materials,
                weight_thickness=weight_thickness,
                weight_materials=weight_materials,
                weight_lab=weight_lab
            )

            # 反向传播
            loss.backward()
            optimizer_inn.step()

            epoch_train_losses.append(loss.item())
            epoch_losses_thickness.append(loss_thickness.item())
            epoch_losses_materials.append(loss_materials.item())
            epoch_losses_lab.append(loss_lab.item())
            if calculate_accuracy:
                epoch_accuracy.append(average_accuracy.item())

        avg_train_loss = np.mean(epoch_train_losses)
        avg_accuracy = np.mean(epoch_accuracy) if calculate_accuracy else "N/A"
        avg_losses_thickness = np.mean(epoch_losses_thickness)
        avg_losses_materials = np.mean(epoch_losses_materials)
        avg_losses_lab = np.mean(epoch_losses_lab)

        tqdm.write(
            f'Epoch {epoch}/{epochs}: Training Loss: {avg_train_loss:.6f}, ' +
            ('Accuracy: {:.6f}, '.format(avg_accuracy) if calculate_accuracy else '') +
            f'Thickness Loss: {avg_losses_thickness:.6f}, ' +
            f'Materials Loss: {avg_losses_materials:.6f}, LAB Loss: {avg_losses_lab:.6f}')

        record_losses = []

        record_losses.append({
            'Epoch': epoch,
            'Training Loss': avg_train_loss,
            'Accuracy': avg_accuracy,
            'Thickness Loss': avg_losses_thickness,
            'Materials Loss': avg_losses_materials,
            'LAB Loss': avg_losses_lab
        })

    loss_data = pd.DataFrame(record_losses)
    loss_data.to_csv('training_performance-label encode_3.12.csv', index=False)

if __name__ == '__main__':
    main()