import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# FashionMNIST 数据集: 经典的多分类数据集，包含 60,000 个训练样本和 10,000 个测试样本，每个样本是一个 28x28 的灰度图像，类别为 10 种不同类型的服装。
def get_fashion_mnist_dataloaders(
    batch_size, num_workers, test_size=0.2, random_state=42
):
    fashion_mnist = load_digits()
    X = fashion_mnist.data
    y = fashion_mnist.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


# Iris 数据集: 经典的多分类数据集，包含 150 个样本，每个样本有 4 个特征，类别为鸢尾花的 3 种类型。
def get_iris_dataloaders(batch_size, num_workers, test_size=0.2, random_state=42):
    iris = load_iris()
    X = iris.data  # 特征：4 维数值数据
    y = iris.target  # 类别标签：3 类鸢尾花

    scaler = StandardScaler()  # 标准化特征数据
    X = scaler.fit_transform(X)

    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 将 NumPy 数组转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


# Breast Cancer 数据集: 一个二分类数据集，包含 569 个样本，每个样本有 30 个特征，用于预测肿瘤类型（良性或恶性）。
def get_breast_cancer_dataloaders(
    batch_size, num_workers, test_size=0.2, random_state=42
):
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data  # 特征：30 维数值数据
    y = breast_cancer.target  # 类别标签：2 类（0 表示恶性，1 表示良性）

    scaler = StandardScaler()  # 标准化特征数据
    X = scaler.fit_transform(X)

    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 将 NumPy 数组转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


# Digits 数据集: 一个多分类的手写数字识别数据集，包含 1797 个 8x8 像素的图像，每个样本是 0-9 之间的数字。
def get_digits_dataloaders(batch_size, num_workers, test_size=0.2, random_state=42):
    digits = load_digits()
    X = digits.images  # 每个图像为 8x8 像素
    y = digits.target  # 类别标签：数字 0-9

    X = X.reshape(X.shape[0], -1)  # 展平图像，使每个图像成为 64 维的向量

    scaler = StandardScaler()  # 标准化特征数据
    X = scaler.fit_transform(X)

    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 将 NumPy 数组转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


# Boston Housing 数据集: 一个回归数据集，包含 506 个样本，每个样本有 13 个特征，目标是预测波士顿房价的中位数。
def get_boston_dataloaders(batch_size, num_workers, test_size=0.2, random_state=42):
    boston = load_boston()
    X = boston.data  # 特征：13 维数值数据
    y = boston.target  # 目标值：房价中位数

    scaler = StandardScaler()  # 标准化特征数据
    X = scaler.fit_transform(X)

    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 将 NumPy 数组转换为 PyTorch 张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


# 数据加载器工厂方法：根据数据集名称返回相应的数据加载器
def get_dataloaders(dataset_name, batch_size, num_workers, data_dir="./data"):
    if dataset_name == "Iris":
        return get_iris_dataloaders(batch_size, num_workers)
    elif dataset_name == "BreastCancer":
        return get_breast_cancer_dataloaders(batch_size, num_workers)
    elif dataset_name == "Digits":
        return get_digits_dataloaders(batch_size, num_workers)
    elif dataset_name == "BostonHousing":
        return get_boston_dataloaders(batch_size, num_workers)
    elif dataset_name == "FashionMNIST":
        return get_fashion_mnist_dataloaders(batch_size, num_workers, data_dir)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
