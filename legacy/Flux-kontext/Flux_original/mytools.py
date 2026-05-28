import pandas as pd
import torch
import os
import numpy as np

def filter_diff_points_pure_torch(diff_indices, image_width, image_height, 
                                   radius=10, min_neighbors=50, min_cluster_size=100):
    """
    完全使用PyTorch实现，不依赖numpy和sklearn
    
    参数:
        diff_indices: torch.Tensor，一维索引
        image_width: 图像宽度
        image_height: 图像高度
        radius: 邻域半径
        min_neighbors: 最小邻居数
        min_cluster_size: 最小聚类大小
    """
    if len(diff_indices) == 0:
        return torch.tensor([])
    
    device = diff_indices.device
    
    # 转换为二维坐标
    y = diff_indices // image_width
    x = diff_indices % image_width
    points = torch.stack([x, y], dim=1).float()
    
    # 计算距离矩阵
    dist_matrix = torch.cdist(points, points)
    
    # 找出每个点的邻居（距离小于radius）
    neighbors = dist_matrix < radius
    neighbor_counts = neighbors.sum(dim=1)
    
    # 只保留邻居数足够的点
    dense_mask = neighbor_counts >= min_neighbors
    dense_indices = torch.where(dense_mask)[0]
    
    if len(dense_indices) == 0:
        return torch.tensor([])
    
    # 简单连通性分析：使用并查集思想
    dense_points = points[dense_mask]
    dense_neighbors = neighbors[dense_mask][:, dense_mask]
    
    # 找最大连通分量
    n_dense = len(dense_indices)
    visited = torch.zeros(n_dense, dtype=torch.bool, device=device)
    clusters = []
    
    for i in range(n_dense):
        if visited[i]:
            continue
        
        # BFS找连通分量
        cluster = [i]
        queue = [i]
        visited[i] = True
        
        while queue:
            curr = queue.pop(0)
            neighbors_curr = torch.where(dense_neighbors[curr])[0]
            
            for neighbor in neighbors_curr:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    cluster.append(neighbor.item())
                    queue.append(neighbor.item())
        
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    
    if not clusters:
        return torch.tensor([])
    
    # 保留所有大聚类
    valid_indices = []
    for cluster in clusters:
        valid_indices.extend([dense_indices[i].item() for i in cluster])
    
    valid_indices = torch.tensor(valid_indices, device=device)
    return diff_indices[valid_indices]

def get_key_token_indices(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    计算两个tensor中对应行向量的余弦相似度
    
    参数:
        tensor1: shape (n, d)
        tensor2: shape (n, d)
        device: 指定运行的GPU设备（如torch.device("cuda:0")），默认为None（自动选择）
    返回:
        similarities: shape (n,), 每一行的余弦相似度
    """
    device = tensor2.device
    tensor1 = tensor1.to(device)
    assert tensor1.shape == tensor2.shape, "两个tensor的shape必须相同"
    
    # 计算每行的点积
    dot_product = (tensor1 * tensor2).sum(dim=1)
    
    # 计算每行的范数
    norm1 = tensor1.norm(p=2, dim=1)
    norm2 = tensor2.norm(p=2, dim=1)
    
    # 防止除零
    eps = 1e-8
    similarities = dot_product / (norm1 * norm2 + eps)
    
    # 将相似度归一化到[0, 1]范围
    normalized_similarities = (similarities + 1) / 2

    # 找出相似度小于阈值的布尔掩码 (mask)
    mask = normalized_similarities < threshold
    # 从掩码中获取索引ID
    indices = mask.nonzero(as_tuple=True)[0]
    return indices

def replace_tensor_with_indices(original_tensor, replacement_tensor, csv_file_path, index_column=0, offset=0):
    """
    根据CSV文件中的索引，替换tensor中的向量
    
    Args:
        original_tensor: 原始tensor [1,4070,64]
        replacement_tensor: 用于替换的tensor [1,4070,64]  
        csv_file_path: CSV文件路径
        index_column: CSV文件中索引列的位置(默认第0列)
    
    Returns:
        result_tensor: 处理后的tensor [1,4070,64]
    """
    
    # 检查tensor维度
    # assert original_tensor.shape == replacement_tensor.shape, f"原始tensor维度为{original_tensor.shape}，替换tensor维度为{replacement_tensor.shape}"
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        # 获取索引列（假设索引在第一列，列名可能不同）
        indices = df.iloc[:, index_column].values
    except Exception as e:
        print(f"读取CSV文件出错: {e}")
        return None
    
    # 验证索引范围
    indices = indices[~pd.isna(indices)]  # 去除NaN值
    indices = (indices + offset).astype(int)  # 转换为整数
    valid_indices = indices[(indices >= offset) & (indices < replacement_tensor.shape[1] + offset)]
    
    if len(valid_indices) != len(indices):
        print(f"警告: 发现{len(indices) - len(valid_indices)}个无效索引（超出0-4014范围）")
    
    # print(f"从CSV读取到{len(valid_indices)}个有效索引")
    
    # 创建结果tensor，初始为replacement_tensor的副本
    result_tensor = replacement_tensor.clone()
    
    # 将指定索引位置的向量替换为原始tensor中的向量
    # print(f"original_tensor.shape: {original_tensor.shape}, replacement_tensor.shape: {replacement_tensor.shape}, result_tensor.shape: {result_tensor.shape}")
    if len(valid_indices) > 0:
        result_tensor[0, valid_indices - offset, :] = original_tensor[0, valid_indices, :]
    
    
    return result_tensor

def read_indices_from_csv(csv_path: str, index_col_name: str, offset: int = 0):
    """
    从CSV文件中读取索引，并返回一个PyTorch LongTensor。

    参数:
        csv_path (str): 包含索引的CSV文件的路径。
        index_col_name (str): CSV文件中包含索引的列名。
        offset (int): 索引偏移量，默认为0。

    返回:
        torch.Tensor: 包含索引的LongTensor。如果发生错误，则返回 None。
    """
    try:
        df = pd.read_csv(csv_path)
        if index_col_name not in df.columns:
            print(f"错误: 列名 '{index_col_name}' 在CSV文件中未找到。")
            return None
        
        indices_list = df[index_col_name].tolist()
        
        # 检查索引是否在有效范围内
        if not indices_list or min(indices_list) < 0:
            print(f"错误: CSV中的一个或多个索引小于0。")
            return None
    except FileNotFoundError:
        print(f"错误: 文件 '{csv_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取CSV或处理索引时发生错误: {e}")
        return None
            
    # 将索引列表转换为Tensor
    indices_tensor = torch.tensor(indices_list, dtype=torch.int32) + offset
    print(f"成功读取 {len(indices_tensor)} 个索引。")
    return indices_tensor

def extract_important_tensor_part_by_csv(
    tensor_to_process1: torch.Tensor, 
    tensor_to_process2: torch.Tensor,
    csv_path: str, 
    index_col_name: str,
    offset: int = 0
):
    """
    根据CSV文件中的索引，沿第二个维度从一个三维张量中提取指定部分。

    参数:
        tensor_to_process1 (torch.Tensor): 需要被处理的 [B, H, N, D] 格式的4维张量。
        tensor_to_process2 (torch.Tensor): 需要被处理的 [B, H, N, D, D] 格式的5维张量。
        csv_path (str): 包含索引的CSV文件的路径。
        index_col_name (str): CSV文件中包含索引的列名。

    返回:
        torch.Tensor: 一个新的张量，包含了所有被提取出的数据。
                      如果发生错误，则返回 None。
    """
    # 1. 验证输入是否为三维张量
    if not isinstance(tensor_to_process1, torch.Tensor) or tensor_to_process1.dim() != 4:
        print("错误: tensor_to_process1必须是一个4维的 PyTorch 张量。")
        return None
    
    if not isinstance(tensor_to_process2, torch.Tensor) or tensor_to_process2.dim() != 6:
        print("错误: tensor_to_process2必须是一个6维的 PyTorch 张量。")
        return None

    print(f"--- 开始从张量 (形状: {tensor_to_process1.shape}) 中提取数据 ---")
    print(f"--- 索引来源: '{csv_path}' ---")

    # 2. 从CSV文件读取索引
    try:
        df = pd.read_csv(csv_path)
        if index_col_name not in df.columns:
            print(f"错误: 列名 '{index_col_name}' 在CSV文件中未找到。")
            return None
        
        indices_list = df[index_col_name].tolist()
        
        # 检查索引是否在有效范围内
        max_index = tensor_to_process1.shape[2] - 1
        if not indices_list or max(indices_list) > max_index or min(indices_list) < 0:
            print(f"错误: CSV中的一个或多个索引超出了张量第二维度的有效范围 [0, {max_index}]。")
            return None
            
        # 将索引列表转换为LongTensor，这是torch.index_select所要求的
        indices_to_extract = torch.tensor(indices_list, dtype=torch.long).to(tensor_to_process1.device)
        print(f"成功读取 {len(indices_to_extract)} 个待提取索引。")

    except FileNotFoundError:
        print(f"错误: 文件 '{csv_path}' 未找到。")
        return None
    except Exception as e:
        print(f"读取CSV或处理索引时发生错误: {e}")
        return None

    # 3. 使用 torch.index_select 沿维度1高效地提取数据
    extracted_tensor1 = torch.index_select(tensor_to_process1, 2, indices_to_extract)
    extracted_tensor2 = torch.index_select(tensor_to_process2, 2, indices_to_extract)
    
    print("提取完成。")
    print(f"提取出的新张量q的形状: {extracted_tensor1.shape}")
    print(f"提取出的新张量pe_q的形状: {extracted_tensor2.shape}")

    return extracted_tensor1, extracted_tensor2


