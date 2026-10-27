import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import requests
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import torch
import torch.nn as nn


# ==================== 模糊搜索功能 ====================
def levenshtein_distance(s1, s2):
    """计算两个字符串的编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # insertions/deletions/substitutions
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(previous_row[j + 1] + 1, current_row[j] + 1, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def fuzzy_search_molecule(user_input, threshold=0.7):
    """
    模糊搜索分子，返回相似的分子列表
    threshold: 相似度阈值 (0-1)，越高越严格
    """
    user_input = user_input.strip().lower()
    if not user_input:
        return []
    
    suggestions = []
    
    # 搜索 MOLECULE_MAP 中的中文名和英文名
    for name, smiles in MOLECULE_MAP.items():
        if isinstance(name, str):
            # 计算相似度
            distance = levenshtein_distance(user_input, name.lower())
            max_distance = int(len(name) * (1 - threshold))
            
            if distance <= max_distance:
                similarity = 1 - distance / max(len(name), len(user_input))
                suggestions.append({
                    'name': name,
                    'smiles': smiles,
                    'similarity': similarity,
                    'type': 'molecule_map'
                })
    
    # 搜索 CHEMICAL_FORMULA_MAP 中的化学式
    for formula, smiles_list in CHEMICAL_FORMULA_MAP.items():
        distance = levenshtein_distance(user_input, formula.lower())
        max_distance = int(len(formula) * (1 - threshold))
        
        if distance <= max_distance:
            similarity = 1 - distance / max(len(formula), len(user_input))
            suggestions.append({
                'name': formula,
                'smiles': smiles_list[0],  # 只取第一个 SMILES
                'similarity': similarity,
                'type': 'chemical_formula'
            })
    
    # 按相似度排序
    suggestions.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 返回相似度最高的前 5 个
    return suggestions[:5]


def get_solubility_description(result, user_input):
    """
    根据预测的 LogS 值解释水溶解度
    result: 预测结果字典，包含 'logS' 键
    user_input: 用户输入的分子名称
    """
    if isinstance(result, dict):
        logS = result.get('logS', 0)
    else:
        # 如果 result 是浮点数（旧格式），直接使用
        logS = result
    
    # 根据 LogS 值解释水溶解度
    if logS > 0:
        description = f"**{user_input}** 的 LogS = {logS:.4f}，**易溶于水**"
    elif logS >= -1:
        description = f"**{user_input}** 的 LogS = {logS:.4f}，**可溶于水**"
    elif logS >= -2:
        description = f"**{user_input}** 的 LogS = {logS:.4f}，**微溶于水**"
    elif logS >= -3:
        description = f"**{user_input}** 的 LogS = {logS:.4f}，**极微溶于水**"
    else:
        description = f"**{user_input}** 的 LogS = {logS:.4f}，**几乎不溶于水**"
    
    return description

# ==================== 模型文件下载 ====================
def download_model_file(filename, url):
    """下载模型文件"""
    # 检查文件是否已存在
    if os.path.exists(filename):
        return True
    
    try:
        print(f"正在下载 {filename}...")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # 创建目录
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 下载文件
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ {filename} 下载完成")
        return True
    except Exception as e:
        print(f"✗ {filename} 下载失败: {e}")
        return False


def ensure_model_files():
    """确保模型文件存在，如果不存在则下载"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 水溶解度模型文件
    solubility_model_dir = os.path.join(script_dir, 'solubility_models')
    solubility_model_file = os.path.join(solubility_model_dir, 'logs_best_model_v8_optimized.pkl')
    
    if not os.path.exists(solubility_model_file):
        # GitHub Release 下载链接
        model_url = 'https://github.com/jasonxu486-a11y/molecular-property-prediction/releases/download/附件/logs_best_model_v8_optimized.pkl'
        download_model_file(solubility_model_file, model_url)


# ==================== 水溶解度预测模型 ====================
class SolubilityPredictor:
    """水溶解度预测器 - 使用本地模型文件"""
    
    def __init__(self, model_dir=None):
        # 如果没有指定 model_dir，使用相对路径
        if model_dir is None:
            # 尝试多种可能的路径
            possible_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solubility_models'),
                os.path.join(os.getcwd(), 'solubility_models'),
                'solubility_models',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_dir = path
                    break
            # 如果还是没找到，使用第一个路径（即使不存在）
            if model_dir is None:
                model_dir = possible_paths[0]
        
        self.model_dir = model_dir
        self.models = None
        self.model_names = None
        self.best_weights = None
        self.scaler = None
        self.variance_threshold = None
        self.select_k_best = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """加载水溶解度预测模型"""
        try:
            # 加载特征名称
            feature_path = os.path.join(self.model_dir, 'logs_feature_names_v8_optimized.txt')
            if os.path.exists(feature_path):
                with open(feature_path, 'r', encoding='utf-8') as f:
                    self.feature_names = [line.strip() for line in f if line.strip()]
            
            # 加载集成模型
            model_path = os.path.join(self.model_dir, 'logs_best_model_v8_optimized.pkl')
            if os.path.exists(model_path):
                model_data = pickle.load(open(model_path, 'rb'))
                self.models = model_data['models']
                self.model_names = model_data['model_names']
                self.best_weights = model_data['best_weights']
                # 从模型数据中加载预处理工具
                self.scaler = model_data.get('scaler')
                self.variance_threshold = model_data.get('var_thresh')
                self.select_k_best = model_data.get('selector')
            
            print("✓ 水溶解度预测模型已加载")
        except Exception as e:
            print(f"⚠ 水溶解度预测模型加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    def predict(self, smiles):
        """预测水溶解度"""
        try:
            from rdkit.Chem import AllChem
            from rdkit.Chem import MACCSkeys
            
            # 生成分子描述符
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print("SMILES 解析失败")
                return None
            
            # 138 descriptors (使用与 predict_logs.py 相同的描述符列表)
            desc_vals = []
            for d in [
                Descriptors.MolWt, Descriptors.MolLogP, Descriptors.MolMR, Descriptors.TPSA,
                Descriptors.HeavyAtomCount, Descriptors.HeavyAtomMolWt,
                Descriptors.RingCount, Descriptors.NumAromaticRings, Descriptors.NumAliphaticRings,
                Descriptors.NumHAcceptors, Descriptors.NumHDonors, Descriptors.NumRotatableBonds,
                Descriptors.NumHeteroatoms, Descriptors.NumHeterocycles,
                Descriptors.NumAromaticCarbocycles, Descriptors.NumAromaticHeterocycles,
                Descriptors.NumAliphaticCarbocycles, Descriptors.NumAliphaticHeterocycles,
                Descriptors.NumSaturatedRings, Descriptors.NumSaturatedHeterocycles,
                Descriptors.NumAmideBonds, Descriptors.NumAtomStereoCenters,
                Descriptors.NumBridgeheadAtoms, Descriptors.ExactMolWt,
                Descriptors.EState_VSA1, Descriptors.EState_VSA2, Descriptors.EState_VSA3,
                Descriptors.EState_VSA4, Descriptors.EState_VSA5, Descriptors.EState_VSA6,
                Descriptors.EState_VSA7, Descriptors.EState_VSA8,
                Descriptors.EState_VSA9, Descriptors.EState_VSA10,
                Descriptors.VSA_EState1, Descriptors.VSA_EState2, Descriptors.VSA_EState3,
                Descriptors.VSA_EState4, Descriptors.VSA_EState5,
                Descriptors.VSA_EState6, Descriptors.VSA_EState7, Descriptors.VSA_EState8,
                Descriptors.PEOE_VSA1, Descriptors.PEOE_VSA2, Descriptors.PEOE_VSA3,
                Descriptors.PEOE_VSA4, Descriptors.PEOE_VSA5,
                Descriptors.PEOE_VSA6, Descriptors.PEOE_VSA7, Descriptors.PEOE_VSA8,
                Descriptors.SlogP_VSA1, Descriptors.SlogP_VSA2, Descriptors.SlogP_VSA3,
                Descriptors.SlogP_VSA4, Descriptors.SlogP_VSA5,
                Descriptors.SlogP_VSA6,
                Descriptors.Chi0n, Descriptors.Chi1n, Descriptors.Chi2n, Descriptors.Chi3n,
                Descriptors.Chi0v, Descriptors.Chi1v, Descriptors.Chi2v, Descriptors.Chi3v,
                Descriptors.Chi0, Descriptors.Chi1,
                Descriptors.BCUT2D_CHGHI, Descriptors.BCUT2D_CHGLO,
                Descriptors.BCUT2D_LOGPHI, Descriptors.BCUT2D_LOGPLOW,
                Descriptors.BCUT2D_MWHI, Descriptors.BCUT2D_MWLOW,
                Descriptors.BCUT2D_MRHI, Descriptors.BCUT2D_MRLOW,
                Descriptors.AUTOCORR2D_1, Descriptors.AUTOCORR2D_2, Descriptors.AUTOCORR2D_3,
                Descriptors.AUTOCORR2D_4, Descriptors.AUTOCORR2D_5, Descriptors.AUTOCORR2D_6,
                Descriptors.AUTOCORR2D_7, Descriptors.AUTOCORR2D_8, Descriptors.AUTOCORR2D_9,
                Descriptors.AUTOCORR2D_10, Descriptors.AUTOCORR2D_11, Descriptors.AUTOCORR2D_12,
                Descriptors.AUTOCORR2D_13, Descriptors.AUTOCORR2D_14, Descriptors.AUTOCORR2D_15,
                Descriptors.AUTOCORR2D_16, Descriptors.AUTOCORR2D_17, Descriptors.AUTOCORR2D_18,
                Descriptors.AUTOCORR2D_19, Descriptors.AUTOCORR2D_20,
                Descriptors.AUTOCORR2D_21, Descriptors.AUTOCORR2D_22, Descriptors.AUTOCORR2D_23,
                Descriptors.AUTOCORR2D_24, Descriptors.AUTOCORR2D_25, Descriptors.AUTOCORR2D_26,
                Descriptors.AUTOCORR2D_27, Descriptors.AUTOCORR2D_28, Descriptors.AUTOCORR2D_29,
                Descriptors.AUTOCORR2D_30,
                Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge,
                Descriptors.MaxAbsPartialCharge, Descriptors.MinAbsPartialCharge,
                Descriptors.MaxEStateIndex, Descriptors.MinEStateIndex,
                Descriptors.MaxAbsEStateIndex, Descriptors.MinAbsEStateIndex,
                Descriptors.BertzCT, Descriptors.HallKierAlpha,
                Descriptors.FractionCSP3, Descriptors.LabuteASA,
                Descriptors.Ipc, Descriptors.BalabanJ,
                Descriptors.fr_Al_OH, Descriptors.fr_Ar_OH,
                Descriptors.fr_COO, Descriptors.fr_COO2,
                Descriptors.fr_Ar_N, Descriptors.fr_NH0, Descriptors.fr_NH1, Descriptors.fr_NH2,
                Descriptors.fr_ester, Descriptors.fr_ether,
                Descriptors.fr_amide, Descriptors.fr_ketone,
                Descriptors.fr_halogen, Descriptors.fr_nitro,
                Descriptors.fr_aldehyde, Descriptors.fr_thiophene,
                Descriptors.fr_furan, Descriptors.fr_pyridine,
                Descriptors.fr_benzene, Descriptors.fr_phenol,
            ]:
                try:
                    desc_vals.append(float(d(mol)))
                except:
                    desc_vals.append(0.0)
            
            # Fingerprints
            ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4096)
            ecfp4_arr = np.array([int(x) for x in ecfp4], dtype=np.float64)
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_arr = np.array([int(x) for x in maccs], dtype=np.float64)
            atompair = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=881)
            atompair_arr = np.array([int(x) for x in atompair], dtype=np.float64)
            
            # 合并所有特征
            X = np.nan_to_num(np.concatenate([desc_vals, ecfp4_arr, maccs_arr, atompair_arr]), nan=0.0, posinf=0.0, neginf=0.0)
            
            # 移除常数特征 (与 predict_logs.py 保持一致)
            const_indices = [4234, 4235, 4236, 4238, 5281]
            keep = np.array([j for j in range(len(X)) if j not in const_indices])
            X = X[keep]
            
            # 预处理
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            X_var = self.variance_threshold.transform(X_scaled)
            X_final = self.select_k_best.transform(X_var)
            
            # 预测
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(X_final)[0]
                predictions.append(pred)
            
            # 加权平均
            weight_dict = dict(zip(self.model_names, self.best_weights))
            final_pred = sum(predictions[i] * weight_dict[name] for i, name in enumerate(self.models.keys()))
            
            return final_pred
            
        except Exception as e:
            print(f"预测出错: {e}")
            import traceback
            traceback.print_exc()
            return None


# ==================== 沸点预测模型 ====================
class BoilingPointPredictor:
    def __init__(self, model_dir=None):
        # 如果没有指定 model_dir，使用相对路径
        if model_dir is None:
            # 尝试多种可能的路径
            possible_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boiling_models'),
                os.path.join(os.getcwd(), 'boiling_models'),
                'boiling_models',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_dir = path
                    print(f"找到模型目录: {model_dir}")
                    break
            # 如果还是没找到，使用第一个路径（即使不存在）
            if model_dir is None:
                model_dir = possible_paths[0]
                print(f"未找到模型目录，使用默认路径: {model_dir}")
        
        self.model_dir = model_dir
        self.models = {}  # 存储5个fold的模型
        self.char2idx = {}
        self.meta = {}
        self._load_model()
    
    def _load_model(self):
        """加载沸点预测 LSTM 模型"""
        try:
            import torch
            import torch.nn as nn
            
            # 加载 smiles_cache
            cache_path = os.path.join(self.model_dir, 'smiles_cache.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.smiles_cache = pickle.load(f)
            
            # 加载 clean_data
            data_path = os.path.join(self.model_dir, 'clean_data.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, encoding='utf-8-sig')
                df['boiling point'] = pd.to_numeric(df['boiling point'], errors='coerce')
                df = df[df['boiling point'] < 1000].reset_index(drop=True)
                y_train = df['boiling point'].values
                self.scaler_y = StandardScaler()
                self.scaler_y.fit(y_train.reshape(-1, 1))
                print(f"Scaler fitted with {len(y_train)} samples")
            else:
                print(f"clean_data.csv not found at {data_path}")
            
            # 定义 char2idx（直接在代码中定义，避免 pickle 文件损坏）
            self.char2idx = {
                1: {'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '.': 6, '/': 7, '1': 8, '2': 9, '3': 10, '4': 11, '=': 12, '@': 13, 'A': 14, 'B': 15, 'C': 16, 'F': 17, 'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25, ']': 26, 'b': 27, 'c': 28, 'i': 29, 'l': 30, 'n': 31, 'o': 32, 'r': 33, 's': 34},
                2: {'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '.': 6, '/': 7, '1': 8, '2': 9, '3': 10, '4': 11, '=': 12, '@': 13, 'A': 14, 'B': 15, 'C': 16, 'F': 17, 'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25, ']': 26, 'b': 27, 'c': 28, 'i': 29, 'l': 30, 'n': 31, 'o': 32, 'r': 33, 's': 34},
                3: {'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '.': 6, '/': 7, '1': 8, '2': 9, '3': 10, '4': 11, '=': 12, '@': 13, 'A': 14, 'B': 15, 'C': 16, 'F': 17, 'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25, ']': 26, 'b': 27, 'c': 28, 'i': 29, 'l': 30, 'n': 31, 'o': 32, 'r': 33, 's': 34},
                4: {'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '.': 6, '/': 7, '1': 8, '2': 9, '3': 10, '4': 11, '=': 12, '@': 13, 'A': 14, 'B': 15, 'C': 16, 'F': 17, 'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25, ']': 26, 'b': 27, 'c': 28, 'i': 29, 'l': 30, 'n': 31, 'o': 32, 'r': 33, 's': 34},
                5: {'#': 1, '(': 2, ')': 3, '+': 4, '-': 5, '.': 6, '/': 7, '1': 8, '2': 9, '3': 10, '4': 11, '=': 12, '@': 13, 'A': 14, 'B': 15, 'C': 16, 'F': 17, 'H': 18, 'I': 19, 'N': 20, 'O': 21, 'P': 22, 'S': 23, '[': 24, '\\': 25, ']': 26, 'b': 27, 'c': 28, 'i': 29, 'l': 30, 'n': 31, 'o': 32, 'r': 33, 's': 34},
            }
            
            # 定义 meta（直接在代码中定义，避免 pickle 文件损坏）
            self.meta = {
                1: {'vocab_size': 35, 'max_len': 100},
                2: {'vocab_size': 35, 'max_len': 100},
                3: {'vocab_size': 35, 'max_len': 100},
                4: {'vocab_size': 35, 'max_len': 100},
                5: {'vocab_size': 35, 'max_len': 100},
            }
            
            # 加载模型
            for fold in range(1, 6):
                model_path = os.path.join(self.model_dir, f'best_lstm_fold{fold}.pth')
                
                if os.path.exists(model_path):
                    model = SMILESLSTM(
                        self.meta[fold]['vocab_size'], 
                        embed_dim=64, 
                        hidden_dim=128, 
                        num_layers=2, 
                        dropout=0.3
                    )
                    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
                    model.eval()
                    self.models[fold] = model
            
        except Exception as e:
            print(f"_load_model error: {e}")
            import traceback
            traceback.print_exc()
            pass
    
    def predict(self, formula, fold=None):
        """预测沸点"""
        try:
            if len(self.models) == 0:
                return None
            
            # 如果没有指定 fold，使用第一个可用的 fold
            if fold is None:
                fold = next(iter(self.models.keys()))
            
            # 检查 scaler_y 是否存在
            if not hasattr(self, 'scaler_y'):
                print(f"ERROR: scaler_y not found")
                return None
            
            # 获取 SMILES
            smiles = self._formula_to_smiles(formula)
            if smiles is None:
                print(f"ERROR: formula {formula} not found in formula_map")
                return None
            
            # 获取特征
            meta = self.meta[fold]
            char2idx = self.char2idx[fold]
            encoded = self._encode_smiles(smiles, meta['max_len'], char2idx)
            
            # 预测
            with torch.no_grad():
                encoded_tensor = torch.LongTensor(encoded).unsqueeze(0)
                model = self.models[fold]
                model.eval()
                pred_scaled = model(encoded_tensor).cpu().numpy()[0]
                pred = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            return {
                'K': pred,
                'C': pred - 273.15,
                'F': pred * 9/5 - 459.67,
                'smiles': smiles,
                'formula': formula
            }
        except Exception as e:
            print(f"BoilingPointPredictor.predict error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _formula_to_smiles(self, formula):
        """将化学式转换为 SMILES"""
        formula_map = {
            'H2O': 'O', 'CO2': 'O=C=O', 'CH4': 'C', 'C2H6': 'CC',
            'C2H4': 'C=C', 'C2H2': 'C#C', 'NH3': 'N', 'CH3OH': 'CO',
            'C2H5OH': 'CCO', 'C6H6': 'c1ccccc1', 'C6H5CH3': 'Cc1ccccc1',
            'C3H8': 'CCC', 'C4H10': 'CCCC', 'C5H12': 'CCCCC',
            'C6H14': 'CCCCCC', 'C7H16': 'CCCCCCC', 'C8H18': 'CCCCCCCC',
            'C9H20': 'CCCCCCCCC', 'C10H22': 'CCCCCCCCCC',
            'C2H6O': 'CCO', 'CH4O': 'CO', 'C3H6O': 'CC(=O)C',
            'C2H4O2': 'CC(=O)O', 'C6H6O': 'c1ccccc1O',
            'C3H8O': 'CCCO', 'C4H10O': 'CCCCO',
            'C4H8O2': 'CCOC(=O)C',  # 乙酸乙酯
            'C2H6O2': 'C(C)O',      # 乙二醇
            'C3H8O3': 'C(C(C)O)O',  # 甘油
            'CH2O': 'C=O',          # 甲醛
            'C3H6': 'C=C(C)C',      # 丙烯
            'C4H8': 'CCCC',         # 丁烯
            'C5H10': 'CCCCC',       # 戊烯
            'C6H12': 'CCCCCC',      # 己烯
            'C7H14': 'CCCCCCC',     # 庚烯
            'C8H16': 'CCCCCCCC',    # 辛烯
            'C6H6O': 'c1ccccc1O',   # 苯酚
            'C6H7N': 'c1ccc(N)cc1', # 苯胺
            'C2H2': 'C#C',          # 乙炔
            'C3H4': 'C#CC',         # 丙炔
            'C4H6': 'C#CCCC',       # 丁炔
            'C2H8N2': 'NCCN',       # 乙二胺
            # 简单化学式
            'C': 'C', 'H2': 'HH', 'O2': 'OO', 'N2': 'NN',
            'HF': 'F', 'HCl': 'Cl', 'HBr': 'Br', 'HI': 'I',
            'NO': 'N=O', 'NO2': 'N(=O)=O', 'N2O': 'N=N=O',
            'SO2': 'S(=O)=O', 'SO3': 'S(=O)(=O)=O',
            'H2S': 'S', 'PH3': 'P', 'SiH4': 'Si',
        }
        return formula_map.get(formula)
    
    def _encode_smiles(self, smiles, max_len, char2idx):
        """编码 SMILES"""
        seq = np.zeros(max_len, dtype=int)
        for i, c in enumerate(smiles):
            if i >= max_len:
                break
            seq[i] = char2idx.get(c, 0)
        return seq


# ==================== 水溶解度预测模型 ====================
class SMILESLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        out = lstm_out.mean(dim=1)
        return self.fc(out).squeeze(-1)


# ==================== 分子名称映射 ====================
from sklearn.preprocessing import StandardScaler

MOLECULE_MAP = {
    # 中文名 -> SMILES
    "水": "O",
    "乙醇": "CCO",
    "甲醇": "CO",
    "苯": "c1ccccc1",
    "甲苯": "Cc1ccccc1",
    "乙酸": "CC(=O)O",
    "丙酮": "CC(=O)C",
    "氨": "N",
    "甲醛": "C=O",
    "甲烷": "C",
    "氯甲烷": "CCl",
    "二甲基亚砜": "CS(C)=O",
    "尿素": "NC(=O)N",
    "一氧化氮": "NO",
    "二氧化硫": "O=S=O",
    "过氧化氢": "OO",
    "硝基苯": "c1ccc(N=O)cc1",
    "邻二甲苯": "Cc1c(C)cccc1",
    "对二甲苯": "Cc1ccc(C)cc1",
    "间二甲苯": "Cc1cc(C)ccc1",
    "乙二醇": "C(C)O",
    "甘油": "C(C(C)O)O",
    "葡萄糖": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "咖啡因": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "阿司匹林": "CC(=O)OC1C2C(C=C(C)C)C(C)(C)OC2C(=O)O",
    "胆固醇": "CC(C)CCCC(C)C1CC(C)C2C(C)(C)CC(C)(C)C12",
    "乙酸乙酯": "CCOC(=O)C",
    "乙醚": "CCOCC",
    "氯仿": "C(Cl)Cl",
    "四氯化碳": "C(Cl)Cl",
    "苯酚": "c1ccc(O)cc1",
    "苯胺": "c1ccc(N)cc1",
    "苯甲酸": "c1ccc(C(=O)O)cc1",
    "丙烯": "C=C",
    "丁烯": "CCCC",
    "戊烷": "CCCCC",
    "己烷": "CCCCCC",
    "庚烷": "CCCCCCC",
    "辛烷": "CCCCCCCC",
    "壬烷": "CCCCCCCCC",
    "癸烷": "CCCCCCCCCC",
    "乙炔": "C#C",
    "丙炔": "C#CC",
    "丁炔": "C#CCCC",
    "乙二胺": "NCCN",
    # 英文名 -> SMILES
    "water": "O",
    "ethanol": "CCO",
    "methanol": "CO",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "acetic acid": "CC(=O)O",
    "acetone": "CC(=O)C",
    "ammonia": "N",
    "formaldehyde": "C=O",
    "methane": "C",
    "chloromethane": "CCl",
    "DMSO": "CS(C)=O",
    "urea": "NC(=O)N",
    "nitric oxide": "NO",
    "nitrobenzene": "c1ccc(N=O)cc1",
    "ethylene glycol": "C(C)O",
    "glycerol": "C(C(C)O)O",
    "glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "aspirin": "CC(=O)OC1C2C(C=C(C)C)C(C)(C)OC2C(=O)O",
    "cholesterol": "CC(C)CCCC(C)C1CC(C)C2C(C)(C)CC(C)(C)C12",
    "ethyl acetate": "CCOC(=O)C",
    "diethyl ether": "CCOCC",
    "chloroform": "C(Cl)Cl",
    "carbon tetrachloride": "C(Cl)Cl",
    "phenol": "c1ccc(O)cc1",
    "aniline": "c1ccc(N)cc1",
    "benzoic acid": "c1ccc(C(=O)O)cc1",
    "propene": "C=C",
    "butene": "CCCC",
    "pentane": "CCCCC",
    "hexane": "CCCCCC",
    "heptane": "CCCCCCC",
    "octane": "CCCCCCCC",
    "nonane": "CCCCCCCCC",
    "decane": "CCCCCCCCCC",
    "acetylene": "C#C",
    "propyne": "C#CC",
    "butyne": "C#CCCC",
    "ethylenediamine": "NCCN",
    # CAS No -> SMILES
    "7732-18-5": "O",
    "64-17-5": "CCO",
    "67-56-1": "CO",
    "71-43-2": "c1ccccc1",
    "108-88-3": "Cc1ccccc1",
    "64-19-7": "CC(=O)O",
    "67-64-1": "CC(=O)C",
    "7664-41-7": "N",
    "50-00-0": "C=O",
    "74-82-8": "C",
    "74-87-3": "CCl",
    "67-68-5": "CS(C)=O",
    "57-13-6": "NC(=O)N",
    "10102-43-9": "NO",
}

# ==================== 化学式到 SMILES 的映射（用于化学式搜索）====================
# 从 clean_data.csv 中提取的常见化学式及其对应的 SMILES
# 当用户输入化学式时，会显示所有匹配的异构体供选择
CHEMICAL_FORMULA_MAP = {
    "H2O": ["O"],  # 水
    "CH4": ["C"],  # 甲烷
    "C2H6O": ["CCO"],  # 乙醇
    "CH4O": ["CO"],  # 甲醇
    "C2H6": ["CC"],  # 乙烷
    "C3H8": ["CCC"],  # 丙烷
    "C4H10": ["CCCC", "CC(C)C"],  # 丁烷、异丁烷
    "C6H6": ["c1ccccc1"],  # 苯
    "C7H8": ["Cc1ccccc1"],  # 甲苯
    "CH2O": ["C=O"],  # 甲醛
    "C2H4O": ["CC=O"],  # 乙醛
    "C3H6O": ["CCC(=O)C", "CC(C)C=O", "C(C)C=O"],  # 丙酮等
    "C4H8O2": ["CCOC(=O)C", "CC(C)C(=O)O", "CCC(=O)O"],  # 乙酸乙酯、异丁酸、丁酸
    "C2H4O2": ["CC(=O)O"],  # 乙酸
    "CH2O2": ["C(=O)O"],  # 甲酸
    "C6H12O6": ["C(C1C(C(C(C(O1)O)O)O)O)O"],  # 葡萄糖
    "C8H10": ["Cc1ccc(C)cc1", "Cc1c(C)cccc1", "Cc1cc(C)ccc1"],  # 二甲苯异构体
    "C4H10O": ["CCCCO", "CC(C)CO", "CCOC(C)C", "C(C)(C)CO"],  # 丁醇异构体
    "C3H8O": ["CCCO", "CC(C)CO"],  # 丙醇异构体
    "C5H12": ["CCCCC", "CC(C)CC", "C(C)(C)C"],  # 戊烷异构体
    "C5H10": ["CCCC=C", "CC(C)C=C", "C(C)(C)CC"],  # 戊烯异构体
    "C6H14": ["CCCCCC", "CC(C)CCCC", "CC(C)C(C)CC"],  # 己烷异构体
    "C7H16": ["CCCCCCC", "CC(C)CCCCC", "CC(C)C(C)CCC"],  # 庚烷异构体
    "C8H18": ["CCCCCCCC", "CC(C)CCCCCC", "CC(C)C(C)CCCCC"],  # 辛烷异构体
    "C2H2": ["C#C"],  # 乙炔
    "C2H3N": ["CC#N"],  # 丙炔腈
    "CH3N": ["CN"],  # 甲胺
    "CH5N": ["CN"],  # 甲胺
    "CH4N2O": ["NC(=O)N"],  # 尿素
    "CH2O3": ["C(=O)O"],  # 甲酸
    "C2H2O": ["C=C=O"],  # 乙烯酮
    "C2H4": ["C=C"],  # 乙烯
    "C3H6": ["CCC", "C=C(C)C"],  # 丙烷、丙烯
    "C4H8": ["CCCC", "CC(C)C", "C=C(C)C", "CC=C(C)C"],  # 丁烷、异丁烷、丁烯
    "C6H10": ["c1ccccc1", "C1CCCCC1"],  # 苯、环己烷
    "C6H12": ["C1CCCCC1", "C1CCCC=C1"],  # 环己烷、环己烯
    "C6H14": ["CCCCCC", "CC(C)CCCC", "CC(C)C(C)CC"],  # 己烷异构体
    "C7H14": ["CCCCCC(C)", "CC(C)CCCCC"],  # 庚烷异构体
    "C8H16": ["CCCCCCCC", "CC(C)CCCCCC"],  # 辛烷异构体
    "C9H20": ["CCCCCCCCC", "CC(C)CCCCCCC"],  # 壬烷异构体
    "C10H22": ["CCCCCCCCCC", "CC(C)CCCCCCCC"],  # 癸烷异构体
    "C2H5Cl": ["CCl"],  # 氯乙烷
    "CH3Cl": ["CCl"],  # 氯甲烷
    "CH2Cl2": ["C(Cl)Cl"],  # 二氯甲烷
    "CHCl3": ["C(Cl)(Cl)Cl"],  # 氯仿
    "CCl4": ["C(Cl)(Cl)(Cl)Cl"],  # 四氯化碳
    "CH2F2": ["CF"],  # 二氟甲烷
    "CH3F": ["CF"],  # 氟甲烷
    "CHF3": ["CF"],  # 氟仿
    "CF4": ["CF"],  # 四氟甲烷
    "C2H4F2": ["CC(F)F", "FC(C)F"],  # 1,2-二氟乙烷
    "C2H3F3": ["CC(F)(F)F", "FC(C)(F)F"],  # 1,1,2-三氟乙烷
    "C2H2F4": ["CC(F)(F)F(F)", "FC(C)(F)F(F)"],  # 1,1,2,2-四氟乙烷
    "C2HF5": ["CC(F)(F)F(F)F", "FC(C)(F)F(F)F"],  # 五氟乙烷
    "CH3NO2": ["C[N+](=O)[O-]"],  # 硝基甲烷
    "C2H5NO2": ["CC[N+](=O)[O-]"],  # 硝基乙烷
    "CH3NO": ["C[N+](=O)[O-]"],  # 亚硝基甲烷
    "C2H6N2": ["CC(N)N"],  # 乙二胺
    "CH6N2": ["N(C)C"],  # 二甲胺
    "C2H7N": ["CCN"],  # 乙胺
    "CH5N": ["CN"],  # 甲胺
    "CH4N2S": ["CSC(=N)N"],  # 硫脲
    "CH2NS": ["C(S)N"],  # 硫代甲酰胺
    "C2H4N2": ["CC(N)=N"],  # 乙二酰二胺
    "CH2N2": ["C(N)=N"],  # 重氮甲烷
    "C2H2N2": ["C(N)=N"],  # 重氮乙烷
    "C3H4N2": ["CCC(N)=N", "CC(C)N=N"],  # 丙二酰二胺
    "C4H4N2": ["CCCC(N)=N", "CCC(C)N=N"],  # 丁二酰二胺
    "C5H6N2": ["CCCCC(N)=N", "CCCC(C)N=N"],  # 戊二酰二胺
    "C6H8N2": ["CCCCCC(N)=N", "CCCCC(C)N=N"],  # 己二酰二胺
    "C2H5NO": ["CC(N)=O"],  # 乙酰胺
    "CH3NO": ["C(N)=O"],  # 甲酰胺
    "CH2NO2": ["C(N)(=O)=O"],  # 硝酸甲酯
    "C2H5NO2": ["CC(N)(=O)=O"],  # 硝酸乙酯
    "CH3NO3": ["C(N)(=O)=O"],  # 硝酸甲酯
    "C2H6N2O": ["CC(N)N(O)"],  # 乙二胺氧化物
    "CH4N2O": ["NC(=O)N"],  # 尿素
    "C2H4N2O": ["CC(N)=N(O)"],  # 乙二酰二胺氧化物
    "C3H6N2O": ["CCC(N)=N(O)"],  # 丙二酰二胺氧化物
    "C4H8N2O": ["CCCC(N)=N(O)"],  # 丁二酰二胺氧化物
    "C5H10N2O": ["CCCCC(N)=N(O)"],  # 戊二酰二胺氧化物
    "C6H12N2O": ["CCCCCC(N)=N(O)"],  # 己二酰二胺氧化物
    "C2H6O2": ["C(C)O"],  # 乙二醇
    "CH4O2": ["CO"],  # 甲醇
    "C3H8O2": ["CC(C)O"],  # 异丙醇
    "C4H10O2": ["CCC(C)O"],  # 丁醇
    "C5H12O2": ["CCCC(C)O"],  # 戊醇
    "C6H14O2": ["CCCCC(C)O"],  # 己醇
    "C2H6S": ["CC(S)C"],  # 乙硫醇
    "CH4S": ["CS"],  # 甲硫醇
    "C2H6SO": ["CC(S)(=O)C"],  # 乙亚砜
    "C2H6O2S": ["CC(S)(=O)(=O)C"],  # 乙砜
    "CH4O2S": ["CS(=O)=O"],  # 甲磺酸
    "C2H6O3S": ["CC(S(=O)(=O)O)O"],  # 乙磺酸
    "C3H8O3S": ["CCC(S(=O)(=O)O)O"],  # 丙磺酸
    "C4H10O3S": ["CCCC(S(=O)(=O)O)O"],  # 丁磺酸
    "C5H12O3S": ["CCCCC(S(=O)(=O)O)O"],  # 戊磺酸
    "C6H14O3S": ["CCCCCC(S(=O)(=O)O)O"],  # 己磺酸
}


# ==================== 化学式到中文名映射 ====================
# 用于化学式输入时显示中文名
FORMULA_TO_CHINESE = {
    "H2O": "水",
    "CH4": "甲烷",
    "C2H6O": "乙醇",
    "CH4O": "甲醇",
    "C2H6": "乙烷",
    "C3H8": "丙烷",
    "C4H10": "丁烷",
    "C6H6": "苯",
    "C7H8": "甲苯",
    "CH2O": "甲醛",
    "C2H4O": "乙醛",
    "C3H6O": "丙酮",
    "C4H8O2": "乙酸乙酯",
    "C2H4O2": "乙酸",
    "CH2O2": "甲酸",
    "C6H12O6": "葡萄糖",
    "C8H10": "二甲苯",
    "C4H10O": "丁醇",
    "C3H8O": "丙醇",
    "C5H12": "戊烷",
    "C5H10": "戊烯",
    "C6H14": "己烷",
    "C7H16": "庚烷",
    "C8H18": "辛烷",
    "C2H2": "乙炔",
    "C2H3N": "丙炔腈",
    "CH3N": "甲胺",
    "CH5N": "甲胺",
    "CH4N2O": "尿素",
    "CH2O3": "甲酸",
    "C2H2O": "乙烯酮",
    "C2H4": "乙烯",
    "C3H6": "丙烯",
    "C4H8": "丁烷",
    "C6H10": "环己烷",
    "C6H12": "环己烷",
    "C7H14": "庚烷",
    "C8H16": "辛烷",
    "C9H20": "壬烷",
    "C10H22": "癸烷",
    "C2H5Cl": "氯乙烷",
    "CH3Cl": "氯甲烷",
    "CH2Cl2": "二氯甲烷",
    "CHCl3": "氯仿",
    "CCl4": "四氯化碳",
    "CH2F2": "二氟甲烷",
    "CH3F": "氟甲烷",
    "CHF3": "氟仿",
    "CF4": "四氟甲烷",
    "C2H4F2": "二氟乙烷",
    "C2H3F3": "三氟乙烷",
    "C2H2F4": "四氟乙烷",
    "C2HF5": "五氟乙烷",
    "CH3NO2": "硝基甲烷",
    "C2H5NO2": "硝基乙烷",
    "CH3NO": "亚硝基甲烷",
    "C2H6N2": "乙二胺",
    "CH6N2": "二甲胺",
    "C2H7N": "乙胺",
    "CH5N": "甲胺",
    "CH4N2S": "硫脲",
    "CH2NS": "硫代甲酰胺",
    "C2H4N2": "乙二酰二胺",
    "CH2N2": "重氮甲烷",
    "C2H2N2": "重氮乙烷",
    "C3H4N2": "丙二酰二胺",
    "C4H4N2": "丁二酰二胺",
    "C5H6N2": "戊二酰二胺",
    "C6H8N2": "己二酰二胺",
    "C2H5NO": "乙酰胺",
    "CH3NO": "甲酰胺",
    "CH2NO2": "硝酸甲酯",
    "C2H5NO2": "硝酸乙酯",
    "CH3NO3": "硝酸甲酯",
    "C2H6N2O": "乙二胺氧化物",
    "C2H4N2O": "乙二酰二胺氧化物",
    "C3H6N2O": "丙二酰二胺氧化物",
    "C4H8N2O": "丁二酰二胺氧化物",
    "C5H10N2O": "戊二酰二胺氧化物",
    "C6H12N2O": "己二酰二胺氧化物",
    "C2H6O2": "乙二醇",
    "CH4O2": "甲醇",
    "C3H8O2": "异丙醇",
    "C4H10O2": "丁醇",
    "C5H12O2": "戊醇",
    "C6H14O2": "己醇",
    "C2H6S": "乙硫醇",
    "CH4S": "甲硫醇",
    "C2H6SO": "乙亚砜",
    "C2H6O2S": "乙砜",
    "CH4O2S": "甲磺酸",
    "C2H6O3S": "乙磺酸",
    "C3H8O3S": "丙磺酸",
    "C4H10O3S": "丁磺酸",
    "C5H12O3S": "戊磺酸",
    "C6H14O3S": "己磺酸",
}

# ==================== 中文名到英文名映射 ====================
CHINESE_TO_ENGLISH = {
    "水": "water",
    "乙醇": "ethanol",
    "甲醇": "methanol",
    "苯": "benzene",
    "甲苯": "toluene",
    "乙酸": "acetic acid",
    "丙酮": "acetone",
    "氨": "ammonia",
    "甲醛": "formaldehyde",
    "甲烷": "methane",
    "氯甲烷": "chloromethane",
    "二甲基亚砜": "DMSO",
    "尿素": "urea",
    "一氧化氮": "nitric oxide",
    "二氧化硫": "sulfur dioxide",
    "过氧化氢": "hydrogen peroxide",
    "硝基苯": "nitrobenzene",
    "邻二甲苯": "o-xylene",
    "对二甲苯": "p-xylene",
    "间二甲苯": "m-xylene",
    "乙二醇": "ethylene glycol",
    "甘油": "glycerol",
    "葡萄糖": "glucose",
    "咖啡因": "caffeine",
    "阿司匹林": "aspirin",
    "胆固醇": "cholesterol",
    "乙酸乙酯": "ethyl acetate",
    "乙醚": "diethyl ether",
    "氯仿": "chloroform",
    "四氯化碳": "carbon tetrachloride",
    "苯酚": "phenol",
    "苯胺": "aniline",
    "苯甲酸": "benzoic acid",
    "丙烯": "propene",
    "丁烯": "butene",
    "戊烷": "pentane",
    "己烷": "hexane",
    "庚烷": "heptane",
    "辛烷": "octane",
    "壬烷": "nonane",
    "癸烷": "decane",
    "乙炔": "acetylene",
    "丙炔": "propyne",
    "丁炔": "butyne",
    "乙二胺": "ethylenediamine",
}

# ==================== PubChem API 查询 ====================
def lookup_pubchem(name_or_smiles):
    """
    通过 PubChem API 联网查找分子信息
    参数可以是名称或 SMILES
    返回 (SMILES, 英文名, CAS号) 或 None
    """
    input_str = name_or_smiles.strip()
    if not input_str:
        return None
    
    # 尝试将中文名转换为英文名
    if input_str in CHINESE_TO_ENGLISH:
        input_str = CHINESE_TO_ENGLISH[input_str]
    
    try:
        # 检查是否是 SMILES
        mol = Chem.MolFromSmiles(input_str)
        if mol is not None:
            # 输入是 SMILES，使用 SMILES 查询
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            import urllib.parse
            encoded_smiles = urllib.parse.quote(canonical)
            
            # Step 1: 获取 CID
            url_cid = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/cids/JSON'
            r_cid = requests.get(url_cid, timeout=10)
            if r_cid.status_code != 200:
                return None
            cid = r_cid.json().get('IdentifierList', {}).get('CID', [None])[0]
            if not cid:
                return None
            
            # Step 2: 使用 CID 获取 SMILES
            url_smiles = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES/JSON'
            r_smiles = requests.get(url_smiles, timeout=10)
            if r_smiles.status_code != 200:
                return None
            smiles = r_smiles.json().get('PropertyTable', {}).get('Properties', [{}])[0].get('SMILES', '')
            if not smiles:
                return None
            
            # Step 3: 使用 CID 获取 IUPACName
            url_iupac = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON'
            r_iupac = requests.get(url_iupac, timeout=10)
            iupac = ''
            if r_iupac.status_code == 200:
                iupac = r_iupac.json().get('PropertyTable', {}).get('Properties', [{}])[0].get('IUPACName', '')
            
            # Step 4: 使用 CID 获取 synonyms (包含 CAS)
            url_synonyms = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON'
            r_synonyms = requests.get(url_synonyms, timeout=10)
            cas = ''
            if r_synonyms.status_code == 200:
                synonyms = r_synonyms.json().get('InformationList', {}).get('Information', [{}])[0].get('Synonym', [])
                for syn in synonyms:
                    # CAS号格式: 数字-数字-数字
                    if re.match(r'^\d{1,7}-\d{2}-\d$', syn):
                        cas = syn
                        break
            
            if smiles:
                return (smiles, iupac or input_str, cas or '')
        else:
            # 输入是名称，使用名称查询
            url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{input_str}/property/SMILES,IUPACName/JSON'
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if 'PropertyTable' in data:
                    props = data['PropertyTable']['Properties'][0]
                    smiles = props.get('SMILES', '')
                    if smiles:
                        iupac = props.get('IUPACName', '')
                        # 获取 synonyms (包含 CAS)
                        cid = props.get('CID', '')
                        cas = ''
                        if cid:
                            url_synonyms = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON'
                            r_synonyms = requests.get(url_synonyms, timeout=10)
                            if r_synonyms.status_code == 200:
                                synonyms = r_synonyms.json().get('InformationList', {}).get('Information', [{}])[0].get('Synonym', [])
                                for syn in synonyms:
                                    if re.match(r'^\d{1,7}-\d{2}-\d$', syn):
                                        cas = syn
                                        break
                        return (smiles, iupac or input_str, cas or '')
    except Exception:
        pass
    
    return None


def resolve_molecule_input(user_input):
    """
    解析用户输入，返回包含分子信息的字典或 None
    支持: 中文名、英文名、化学式、SMILES、CAS No
    优先使用本地字典，失败则通过 PubChem API 联网查询
    返回: {'smiles': SMILES, 'chinese_name': 中文名, 'english_name': 英文名, 'cas': CAS号} 或 None
    """
    user_input = user_input.strip()
    
    # 如果输入是SMILES，直接返回
    mol = Chem.MolFromSmiles(user_input)
    if mol is not None:
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        formula = generate_chemical_formula(mol)
        # 尝试从 PubChem 获取更多信息
        pubchem = lookup_pubchem(canonical)
        
        # 尝试从 CHINESE_TO_ENGLISH 反向查找中文名
        chinese_name = ''
        if pubchem:
            en = pubchem[1]
            for cn, en_name in CHINESE_TO_ENGLISH.items():
                if en_name.lower() == en.lower():
                    chinese_name = cn
                    break
        
        if pubchem:
            return {'smiles': canonical, 'chinese_name': chinese_name, 'english_name': pubchem[1], 'cas': pubchem[2], 'formula': formula}
        # 如果 PubChem 查询失败，返回基本的分子信息
        return {'smiles': canonical, 'chinese_name': chinese_name, 'english_name': 'N/A', 'cas': 'N/A', 'formula': formula}
    
    # 尝试从映射表中查找
    if user_input in MOLECULE_MAP:
        smiles = MOLECULE_MAP[user_input]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            formula = generate_chemical_formula(mol)
            # 尝试从 PubChem 获取更多信息
            pubchem = lookup_pubchem(smiles)
            
            # 判断输入类型并设置 chinese_name
            chinese_name = ''
            # 检查是否是中文名（在 CHINESE_TO_ENGLISH 中的键）
            if user_input in CHINESE_TO_ENGLISH:
                chinese_name = user_input
            # 检查是否是中文名的大小写变体
            elif user_input.lower() in [k.lower() for k in CHINESE_TO_ENGLISH.keys()]:
                for cn in CHINESE_TO_ENGLISH.keys():
                    if cn.lower() == user_input.lower():
                        chinese_name = cn
                        break
            # 检查是否是英文名，如果是则反向查找中文名
            else:
                for cn, en in CHINESE_TO_ENGLISH.items():
                    if en.lower() == user_input.lower():
                        chinese_name = cn
                        break
            
            if pubchem:
                return {'smiles': smiles, 'chinese_name': chinese_name, 'english_name': pubchem[1], 'cas': pubchem[2], 'formula': formula}
            return {'smiles': smiles, 'chinese_name': chinese_name, 'english_name': '', 'cas': '', 'formula': formula}
    
    # 尝试不区分大小写的英文名匹配
    lower_input = user_input.lower()
    for name, smiles in MOLECULE_MAP.items():
        if isinstance(name, str) and name.lower() == lower_input:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                formula = generate_chemical_formula(mol)
                pubchem = lookup_pubchem(smiles)
                if pubchem:
                    return {'smiles': smiles, 'chinese_name': '', 'english_name': pubchem[1], 'cas': pubchem[2], 'formula': formula}
                return {'smiles': smiles, 'chinese_name': '', 'english_name': '', 'cas': '', 'formula': formula}
    
    # 尝试CAS No匹配
    if re.match(r'^\d{1,7}-\d{2}-\d$', user_input):
        if user_input in MOLECULE_MAP:
            smiles = MOLECULE_MAP[user_input]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                formula = generate_chemical_formula(mol)
                pubchem = lookup_pubchem(smiles)
                
                # 从 MOLECULE_MAP 中查找中文名（中文名键在 CHINESE_TO_ENGLISH 中）
                chinese_name = ''
                for name, mol_smiles in MOLECULE_MAP.items():
                    if isinstance(name, str) and name in CHINESE_TO_ENGLISH and mol_smiles == smiles:
                        chinese_name = name
                        break
                
                # 如果没找到，尝试从 SMILES 反向查找
                if not chinese_name:
                    for cn, en in CHINESE_TO_ENGLISH.items():
                        pubchem_result = lookup_pubchem(en)
                        if pubchem_result and pubchem_result[0] == smiles:
                            chinese_name = cn
                            break
                
                if pubchem:
                    return {'smiles': smiles, 'chinese_name': chinese_name, 'english_name': pubchem[1], 'cas': user_input, 'formula': formula}
                return {'smiles': smiles, 'chinese_name': chinese_name, 'english_name': '', 'cas': user_input, 'formula': formula}
    
    # 尝试化学式匹配
    formula_input = user_input.replace(' ', '')
    if re.match(r'^[A-Z][a-z]?\d*([A-Z][a-z]?\d*)*$', formula_input):
        if formula_input in CHEMICAL_FORMULA_MAP:
            smiles_list = CHEMICAL_FORMULA_MAP[formula_input]
            if len(smiles_list) == 1:
                mol = Chem.MolFromSmiles(smiles_list[0])
                if mol:
                    formula = generate_chemical_formula(mol)
                    # 从 FORMULA_TO_CHINESE 中查找中文名
                    chinese_name = FORMULA_TO_CHINESE.get(formula_input, '')
                    
                    pubchem = lookup_pubchem(smiles_list[0])
                    if pubchem:
                        return {'smiles': smiles_list[0], 'chinese_name': chinese_name, 'english_name': pubchem[1], 'cas': pubchem[2], 'formula': formula}
                    return {'smiles': smiles_list[0], 'chinese_name': chinese_name, 'english_name': '', 'cas': '', 'formula': formula}
            else:
                return {'smiles': None, 'chinese_name': '', 'english_name': f'CHEMICAL_FORMULA:{formula_input}', 'cas': smiles_list, 'formula': formula_input, 'is_formula_choice': True}
    
    # 本地字典没找到，尝试通过 PubChem API 联网查询
    pubchem_result = lookup_pubchem(user_input)
    if pubchem_result:
        smiles, en, cas = pubchem_result
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            formula = generate_chemical_formula(mol)
            # 尝试获取中文名
            chinese_name = ''
            for cn, en_name in CHINESE_TO_ENGLISH.items():
                if en_name.lower() == en.lower():
                    chinese_name = cn
                    break
            return {'smiles': smiles, 'chinese_name': chinese_name, 'english_name': en, 'cas': cas, 'formula': formula}
    
    # 如果所有方法都失败，尝试模糊搜索
    suggestions = fuzzy_search_molecule(user_input)
    if suggestions:
        return {'smiles': None, 'chinese_name': '', 'english_name': f'FUZZY_SEARCH:{user_input}', 'cas': suggestions, 'formula': '', 'is_fuzzy_search': True}
    
    return None


def generate_chemical_formula(mol):
    """根据分子生成化学式（如 C2H6O）"""
    from rdkit.Chem import rdMolDescriptors
    return rdMolDescriptors.CalcMolFormula(mol)


# ==================== 沸点预测输入解析 ====================
def get_boiling_point_input(user_input):
    """
    从用户输入中提取化学式或SMILES
    支持: 中文名、英文名、化学式、SMILES、CAS No
    返回: {'input': 原输入, 'formula': 化学式, 'smiles': SMILES} 或 None
    """
    user_input = user_input.strip()
    
    # 如果输入是SMILES，直接返回
    mol = Chem.MolFromSmiles(user_input)
    if mol is not None:
        formula = generate_chemical_formula(mol)
        return {'input': user_input, 'formula': formula, 'smiles': user_input}
    
    # 尝试从映射表中查找
    if user_input in MOLECULE_MAP:
        smiles = MOLECULE_MAP[user_input]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            formula = generate_chemical_formula(mol)
            return {'input': user_input, 'formula': formula, 'smiles': smiles}
    
    # 尝试不区分大小写的英文名匹配
    lower_input = user_input.lower()
    for name, smiles in MOLECULE_MAP.items():
        if isinstance(name, str) and name.lower() == lower_input:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                formula = generate_chemical_formula(mol)
                return {'input': user_input, 'formula': formula, 'smiles': smiles}
    
    # 尝试化学式匹配
    # 首先规范化化学式（移除空格，统一大小写）
    formula_input = user_input.replace(' ', '')
    if re.match(r'^[A-Z][a-z]?\d*([A-Z][a-z]?\d*)*$', formula_input):
        if formula_input in CHEMICAL_FORMULA_MAP:
            smiles_list = CHEMICAL_FORMULA_MAP[formula_input]
            if len(smiles_list) == 1:
                # 只有一个异构体，直接返回
                mol = Chem.MolFromSmiles(smiles_list[0])
                if mol:
                    formula = generate_chemical_formula(mol)
                    return {'input': user_input, 'formula': formula, 'smiles': smiles_list[0]}
            else:
                # 有多个异构体，返回化学式和所有可能的SMILES
                return {'input': user_input, 'formula': formula_input, 'smiles': None, 'smiles_list': smiles_list}
    
    # 本地字典没找到，尝试通过 PubChem API 联网查询
    pubchem_result = lookup_pubchem(user_input)
    if pubchem_result:
        smiles, en, cas = pubchem_result
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            formula = generate_chemical_formula(mol)
            return {'input': user_input, 'formula': formula, 'smiles': smiles}
    
    return None


def get_smiles_from_input(user_input):
    """
    从用户输入中提取SMILES
    支持: 中文名、英文名、SMILES、CAS No
    """
    user_input = user_input.strip()
    
    # 如果输入是SMILES，直接返回
    if Chem.MolFromSmiles(user_input) is not None:
        return user_input
    
    # 尝试从映射表中查找
    if user_input in MOLECULE_MAP:
        return MOLECULE_MAP[user_input]
    
    # 尝试不区分大小写的英文名匹配
    lower_input = user_input.lower()
    for name, smiles in MOLECULE_MAP.items():
        if isinstance(name, str) and name.lower() == lower_input:
            return smiles
    
    # 尝试CAS No匹配
    if re.match(r'^\d{1,7}-\d{2}-\d$', user_input):
        if user_input in MOLECULE_MAP:
            return MOLECULE_MAP[user_input]
    
    # 本地字典没找到，尝试通过 PubChem API 联网查询
    pubchem_result = lookup_pubchem(user_input)
    if pubchem_result:
        smiles, en, cas = pubchem_result
        return smiles
    
    return None


# ==================== 页面显示函数 ====================
def show_solubility_page():
    st.markdown('<h2 class="sub-header">水溶解度预测 (LogS)</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 输入分子")
        st.markdown("支持输入: **中文名**、**英文名**、**化学式**、**SMILES**、**CAS No**")
        
        # 初始化 session state（如果不存在）
        if 'solubility_input_text' not in st.session_state:
            st.session_state['solubility_input_text'] = 'CCO'
        
        # 如果按钮被点击，重置状态
        if st.session_state.get('solubility_button_clicked', False):
            st.session_state['solubility_button_clicked'] = False
        
        # 创建标签页式输入
        input_mode = st.radio(
            "选择输入方式",
            ["SMILES", "化学式", "中文名", "英文名", "CAS No"],
            horizontal=True,
            key="solubility_input_mode"
        )
        
        user_input = None
        
        # 根据输入模式显示对应的文本框
        if input_mode == "SMILES":
            user_input = st.text_input("输入 SMILES", "CCO", help="例如: CCO, c1ccccc1, O")
        elif input_mode == "化学式":
            user_input = st.text_input("输入 化学式", "C2H6O", help="例如: C2H6O, C6H6, H2O")
        elif input_mode == "中文名":
            user_input = st.text_input("输入 中文名", "水", help="例如: 水, 乙醇, 苯, 甲苯")
        elif input_mode == "英文名":
            user_input = st.text_input("输入 英文名", "water", help="例如: water, ethanol, benzene, toluene")
        elif input_mode == "CAS No":
            user_input = st.text_input("输入 CAS No", "7732-18-5", help="例如: 7732-18-5 (水), 64-17-5 (乙醇)")
        
        # 移除调试输出：不再显示 "解析为 SMILES"
        
        if st.button("预测水溶解度", type="primary", use_container_width=True):
            if user_input:
                with st.spinner("正在预测..."):
                    result = predict_solubility(user_input)
                    if result is not None:
                        st.success("预测成功！")
                        
                        # 分子基本信息
                        st.markdown("### 分子基本信息")
                        mol_info_cols = st.columns(2)
                        with mol_info_cols[0]:
                            st.info(f"""
                            **中文名**: {result.get('chinese_name', 'N/A')}  
                            **英文名**: {result.get('english_name', 'N/A')}  
                            **SMILES**: {result.get('smiles', 'N/A')}  
                            **CAS No**: {result.get('cas', 'N/A')}
                            """)
                        with mol_info_cols[1]:
                            st.info(f"""
                            **化学式**: {result.get('formula', 'N/A')}  
                            **LogS 预测值**: {result['logS']:.4f}
                            """)
                        
                        # 预测结果
                        st.markdown("### 预测结果")
                        st.metric("LogS (水溶解度)", f"{result['logS']:.4f}")
                        
                        # 解释 LogS
                        st.markdown("### 结果解释")
                        solubility_desc = get_solubility_description(result['logS'], user_input)
                        st.info(solubility_desc)
                        
                        # 详细信息
                        st.markdown("### 详细信息")
                        st.info(f"""
                        **输入**: {user_input}  
                        **预测模型**: XGBoost + RF + MLP + Stacking (Ridge)  
                        **测试集 R²**: 0.8137  
                        **测试集 MAE**: 0.69  
                        **测试集 RMSE**: 1.01
                        """)
                    else:
                        st.error("预测失败，请检查输入格式是否正确")
    
    with col2:
        st.markdown("### 使用说明")
        st.markdown("""
        #### 输入格式
        - **中文名**: 水, 乙醇, 苯, 甲苯
        - **英文名**: water, ethanol, benzene, toluene
        - **SMILES**: CCO, c1ccccc1, O=C=O
        - **CAS No**: 7732-18-5 (水), 64-17-5 (乙醇)
        
        #### LogS 解释
        - LogS > 0: 易溶于水
        - LogS = -1 ~ 0: 可溶于水
        - LogS = -2 ~ -1: 微溶于水
        - LogS = -3 ~ -2: 极微溶于水
        - LogS < -3: 几乎不溶于水
        
        #### 示例（数据集中的分子）
        这些是模型训练数据集中常见的分子，预测误差较小：
        
        - **乙醇 (CCO)**: LogS ≈ -0.18 (易溶) - R² > 0.95
        - **苯 (c1ccccc1)**: LogS ≈ -2.42 (难溶) - R² > 0.95
        - **甲苯 (Cc1ccccc1)**: LogS ≈ -2.21 (难溶) - R² > 0.95
        - **丙酮 (CC(=O)C)**: LogS ≈ -0.75 (可溶) - R² > 0.95
        - **乙酸 (CC(=O)O)**: LogS ≈ -0.76 (可溶) - R² > 0.95
        
        ⚠️ 注意：水的预测误差较大（模型训练数据中缺少水分子），请谨慎参考。
        """)
        
        st.markdown("### 常用分子库")
        common_molecules = {
            "水 (O)": "O", "乙醇 (CCO)": "CCO", "甲醇 (CO)": "CO",
            "苯 (c1ccccc1)": "c1ccccc1", "甲苯 (Cc1ccccc1)": "Cc1ccccc1", 
            "乙酸 (CC(=O)O)": "CC(=O)O", "丙酮 (CC(=O)C)": "CC(=O)C",
            "氨 (N)": "N", "甲醛 (C=O)": "C=O", "甲烷 (C)": "C",
            "氯甲烷 (CCl)": "CCl", "DMSO (CS(C)=O)": "CS(C)=O",
            "尿素 (NC(=O)N)": "NC(=O)N", "一氧化氮 (NO)": "NO",
            "乙二醇 (C(C)O)": "C(C)O", "甘油 (C(C(C)O)O)": "C(C(C)O)O",
            "葡萄糖": "C(C1C(C(C(C(O1)O)O)O)O)O", "咖啡因": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "乙酸乙酯": "CCOC(=O)C", "乙醚": "CCOCC",
        }
        
        # 分两列显示按钮
        mol_list = list(common_molecules.items())
        col_btn1, col_btn2 = st.columns(2)
        for i, (name, smiles) in enumerate(mol_list):
            with col_btn1 if i % 2 == 0 else col_btn2:
                if st.button(name, key=f"mol_{i}"):
                    # 直接使用 SMILES 进行预测
                    with st.spinner("正在预测..."):
                        result = predict_solubility(smiles)
                        if result is not None:
                            st.success("预测成功！")
                            
                            # 分子基本信息
                            st.markdown("### 分子基本信息")
                            mol_info_cols = st.columns(2)
                            with mol_info_cols[0]:
                                st.info(f"""
                                **中文名**: {result.get('chinese_name', 'N/A')}  
                                **英文名**: {result.get('english_name', 'N/A')}  
                                **SMILES**: {result.get('smiles', 'N/A')}  
                                **CAS No**: {result.get('cas', 'N/A')}
                                """)
                            with mol_info_cols[1]:
                                st.info(f"""
                                **化学式**: {result.get('formula', 'N/A')}  
                                **LogS 预测值**: {result['logS']:.4f}
                                """)
                            
                            # 预测结果
                            st.markdown("### 预测结果")
                            st.metric("LogS (水溶解度)", f"{result['logS']:.4f}")
                            
                            # 解释 LogS
                            st.markdown("### 结果解释")
                            solubility_desc = get_solubility_description(result['logS'], result.get('chinese_name', ''))
                            st.info(solubility_desc)
                        else:
                            st.error("预测失败，请检查输入格式是否正确")


def show_boiling_point_page():
    st.markdown('<h2 class="sub-header">沸点预测</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 输入分子")
        st.markdown("支持输入: **中文名**、**英文名**、**化学式**、**SMILES**、**CAS No**")
        
        # 创建标签页式输入
        input_mode = st.radio(
            "选择输入方式",
            ["SMILES", "化学式", "中文名", "英文名", "CAS No"],
            horizontal=True,
            key="boiling_input_mode"
        )
        
        user_input = None
        
        if input_mode == "SMILES":
            user_input = st.text_input("输入 SMILES", "CCO", help="例如: CCO, c1ccccc1, O")
        elif input_mode == "化学式":
            user_input = st.text_input("输入 化学式", "C2H6O", help="例如: C2H6O, C6H6, H2O")
        elif input_mode == "中文名":
            user_input = st.text_input("输入 中文名", "乙醇", help="例如: 水, 乙醇, 苯, 甲苯")
        elif input_mode == "英文名":
            user_input = st.text_input("输入 英文名", "ethanol", help="例如: water, ethanol, benzene, toluene")
        elif input_mode == "CAS No":
            user_input = st.text_input("输入 CAS No", "64-17-5", help="例如: 7732-18-5 (水), 64-17-5 (乙醇)")
        
        # 显示解析的化学式或SMILES
        if user_input:
            parsed_info = get_boiling_point_input(user_input)
            if parsed_info and parsed_info['input'] != user_input:
                if parsed_info['smiles']:
                    st.info(f"🔍 解析为 SMILES: `{parsed_info['smiles']}`")
                elif parsed_info['formula']:
                    st.info(f"🔍 解析为 化学式: `{parsed_info['formula']}`")
        
        if st.button("预测沸点", type="primary", use_container_width=True):
            if user_input:
                with st.spinner("正在预测..."):
                    result = predict_boiling_point(user_input)
                    if result:
                        # 如果是化学式选择模式
                        if result.get('is_formula_choice'):
                            st.success("找到多个匹配的异构体！")
                            st.markdown("### 请选择具体的异构体")
                            
                            formula = result['formula']
                            smiles_list = result['smiles_list']
                            
                            # 创建选项字典
                            options = {}
                            for smiles in smiles_list:
                                mol = Chem.MolFromSmiles(smiles)
                                if mol:
                                    name = Chem.MolToSmiles(mol, isomericSmiles=True)
                                    options[name] = smiles
                            
                            # 显示选择按钮
                            selected_smiles = st.radio(
                                "选择异构体",
                                list(options.keys()),
                                key=f"formula_choice_{formula}"
                            )
                            
                            if selected_smiles:
                                selected_smiles = options[selected_smiles]
                                st.info(f"已选择: {selected_smiles}")
                                
                                # 使用选中的 SMILES 进行预测
                                mol = Chem.MolFromSmiles(selected_smiles)
                                if mol:
                                    formula = generate_chemical_formula(mol)
                                    
                                    # 沸点模型路径（使用相对路径）
                                    script_dir = os.path.dirname(os.path.abspath(__file__))
                                    model_dir = os.path.join(script_dir, 'boiling_models')
                                    
                                    predictor = BoilingPointPredictor(model_dir)
                                    
                                    result = predictor.predict(formula)
                                    if result:
                                        st.success("预测成功！")
                                        st.markdown("### 预测结果")
                                        
                                        col_a, col_b, col_c = st.columns(3)
                                        with col_a:
                                            st.metric("开尔文 (K)", f"{result['K']:.1f}")
                                        with col_b:
                                            st.metric("摄氏度 (°C)", f"{result['C']:.1f}")
                                        with col_c:
                                            st.metric("华氏度 (°F)", f"{result['F']:.1f}")
                                        
                                        # 详细信息
                                        st.markdown("### 详细信息")
                                        st.info(f"""
                                        **输入**: {user_input}  
                                        **化学式**: {formula}  
                                        **选中的 SMILES**: {selected_smiles}  
                                        **预测方法**: LSTM (5折集成)
                                        """)
                        else:
                            st.success("预测成功！")
                            
                            # 分子基本信息
                            st.markdown("### 分子基本信息")
                            mol_info_cols = st.columns(2)
                            with mol_info_cols[0]:
                                st.info(f"""
                                **中文名**: {result.get('chinese_name', 'N/A')}  
                                **英文名**: {result.get('english_name', 'N/A')}  
                                **SMILES**: {result.get('smiles', 'N/A')}  
                                **CAS No**: {result.get('cas', 'N/A')}
                                """)
                            with mol_info_cols[1]:
                                st.info(f"""
                                **化学式**: {result.get('formula', 'N/A')}  
                                **沸点预测值**: {result['C']:.1f} °C
                                """)
                            
                            # 预测结果
                            st.markdown("### 预测结果")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("开尔文 (K)", f"{result['K']:.1f}")
                            with col_b:
                                st.metric("摄氏度 (°C)", f"{result['C']:.1f}")
                            with col_c:
                                st.metric("华氏度 (°F)", f"{result['F']:.1f}")
                            
                            # 详细信息
                            st.markdown("### 详细信息")
                            st.info(f"""
                            **输入**: {user_input}  
                            **预测方法**: LSTM (5折集成)
                            """)
                    else:
                        st.error("预测失败，请检查输入格式是否正确")
    
    with col2:
        st.markdown("### 使用说明")
        st.markdown("""
        #### 输入格式
        - **中文名**: 水, 乙醇, 苯, 甲苯
        - **英文名**: water, ethanol, benzene, toluene
        - **化学式**: C2H6O, C6H6, H2O
        - **SMILES**: CCO, c1ccccc1, O
        - **CAS No**: 7732-18-5 (水), 64-17-5 (乙醇)
        
        #### 示例分子
        - 乙醇: `C2H6O` 或 `CCO`
        - 苯: `C6H6` 或 `c1ccccc1`
        - 水: `H2O` 或 `O`
        - 甲烷: `CH4` 或 `C`
        
        #### 输出说明
        - **K**: 开尔文温度
        - **°C**: 摄氏度
        - **°F**: 华氏度
        
        #### 模型性能
        - R² = 0.995 (预期)
        - MAE < 5 K (预期)
        """)
        
        st.markdown("### 常用分子库")
        common_molecules = {
            "水 (H2O)": "H2O", "乙醇 (C2H6O)": "C2H6O", "甲醇 (CH4O)": "CH4O",
            "苯 (C6H6)": "C6H6", "甲苯 (C7H8)": "C7H8", "乙酸 (C2H4O2)": "C2H4O2",
            "丙酮 (C3H6O)": "C3H6O", "氨 (NH3)": "NH3", "甲醛 (CH2O)": "CH2O",
            "乙酸乙酯 (C4H8O2)": "C4H8O2",
        }
        
        # 分两列显示按钮
        mol_list = list(common_molecules.items())
        col_btn1, col_btn2 = st.columns(2)
        for i, (name, formula) in enumerate(mol_list):
            with col_btn1 if i % 2 == 0 else col_btn2:
                if st.button(name, key=f"bp_mol_{i}"):
                    # 直接使用化学式进行预测
                    with st.spinner("正在预测..."):
                        result = predict_boiling_point(formula)
                        if result:
                            # 如果是化学式选择模式
                            if result.get('is_formula_choice'):
                                st.success("找到多个匹配的异构体！")
                                st.markdown("### 请选择具体的异构体")
                                
                                formula_val = result['formula']
                                smiles_list = result['smiles_list']
                                
                                # 创建选项字典
                                options = {}
                                for smiles in smiles_list:
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        name = Chem.MolToSmiles(mol, isomericSmiles=True)
                                        options[name] = smiles
                                
                                # 显示选择按钮
                                selected_smiles = st.radio(
                                    "选择异构体",
                                    list(options.keys()),
                                    key=f"formula_choice_{formula_val}"
                                )
                                
                                if selected_smiles:
                                    selected_smiles = options[selected_smiles]
                                    st.info(f"已选择: {selected_smiles}")
                                    
                                    # 使用选中的 SMILES 进行预测
                                    mol = Chem.MolFromSmiles(selected_smiles)
                                    if mol:
                                        formula_val = generate_chemical_formula(mol)
                                        
                                        # 沸点模型路径（使用相对路径）
                                        script_dir = os.path.dirname(os.path.abspath(__file__))
                                        model_dir = os.path.join(script_dir, 'boiling_models')
                                        
                                        predictor = BoilingPointPredictor(model_dir)
                                        
                                        result = predictor.predict(formula_val)
                                        if result:
                                            st.success("预测成功！")
                                            st.markdown("### 预测结果")
                                            
                                            col_a, col_b, col_c = st.columns(3)
                                            with col_a:
                                                st.metric("开尔文 (K)", f"{result['K']:.1f}")
                                            with col_b:
                                                st.metric("摄氏度 (°C)", f"{result['C']:.1f}")
                                            with col_c:
                                                st.metric("华氏度 (°F)", f"{result['F']:.1f}")
                                            
                                            # 分子基本信息
                                            st.markdown("### 分子基本信息")
                                            mol_info_cols = st.columns(2)
                                            with mol_info_cols[0]:
                                                st.info(f"""
                                                **中文名**: {result.get('chinese_name', 'N/A')}  
                                                **英文名**: {result.get('english_name', 'N/A')}  
                                                **SMILES**: {result.get('smiles', 'N/A')}  
                                                **CAS No**: {result.get('cas', 'N/A')}
                                                """)
                                            with mol_info_cols[1]:
                                                st.info(f"""
                                                **化学式**: {result.get('formula', 'N/A')}  
                                                **沸点预测值**: {result['C']:.1f} °C
                                                """)
                                            
                                            # 详细信息
                                            st.markdown("### 详细信息")
                                            st.info(f"""
                                            **输入**: {formula}  
                                            **预测方法**: LSTM (5折集成)
                                            """)
                            else:
                                st.success("预测成功！")
                                st.markdown("### 预测结果")
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("开尔文 (K)", f"{result['K']:.1f}")
                                with col_b:
                                    st.metric("摄氏度 (°C)", f"{result['C']:.1f}")
                                with col_c:
                                    st.metric("华氏度 (°F)", f"{result['F']:.1f}")
                                
                                # 分子基本信息
                                st.markdown("### 分子基本信息")
                                mol_info_cols = st.columns(2)
                                with mol_info_cols[0]:
                                    st.info(f"""
                                    **中文名**: {result.get('chinese_name', 'N/A')}  
                                    **英文名**: {result.get('english_name', 'N/A')}  
                                    **SMILES**: {result.get('smiles', 'N/A')}  
                                    **CAS No**: {result.get('cas', 'N/A')}
                                    """)
                                with mol_info_cols[1]:
                                    st.info(f"""
                                    **化学式**: {result.get('formula', 'N/A')}  
                                    **沸点预测值**: {result['C']:.1f} °C
                                    """)
                                
                                # 详细信息
                                st.markdown("### 详细信息")
                                st.info(f"""
                                **输入**: {formula}  
                                **预测方法**: LSTM (5折集成)
                                """)
                        else:
                            st.error("预测失败，请检查输入格式是否正确")


# ==================== 预测函数 ====================
def predict_boiling_point(user_input):
    """预测沸点"""
    try:
        user_input = user_input.strip()
        
        # 首先检查输入是否可能是SMILES
        # 如果输入只包含大写字母、小写字母和数字，很可能是化学式而不是SMILES
        # SMILES 通常包含特殊字符如 =, #, @, [, ], (, ), - 等
        is_likely_chemical_formula = bool(re.match(r'^[A-Z][a-z]?[\d]*$', user_input))
        
        mol = None
        if not is_likely_chemical_formula:
            # 只有当输入不像是化学式时，才尝试解析为 SMILES
            mol = Chem.MolFromSmiles(user_input)
        
        if mol is not None:
            # 输入是SMILES，直接生成化学式
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            formula = generate_chemical_formula(mol)
            # 从 resolve_molecule_input 获取分子信息
            result_info = resolve_molecule_input(smiles)
            if result_info:
                chinese_name = result_info.get('chinese_name', '')
                en = result_info.get('english_name', '')
                cas = result_info.get('cas', '')
            else:
                chinese_name = ''
                en = ''
                cas = ''
        else:
            # 解析用户输入（中文名、英文名、化学式、CAS No）
            result = resolve_molecule_input(user_input)
            if result is None:
                return None
            
            # 检查是否是化学式选择模式
            if result.get('is_formula_choice'):
                return {'is_formula_choice': True, 'formula': result['formula'], 'smiles_list': result['cas']}
            
            # 检查是否是模糊搜索模式
            if result.get('is_fuzzy_search'):
                return {'is_fuzzy_search': True, 'user_query': result['english_name'].replace('FUZZY_SEARCH:', ''), 'suggestions': result['cas']}
            
            smiles = result['smiles']
            chinese_name = result.get('chinese_name', '')
            en = result.get('english_name', '')
            cas = result.get('cas', '')
            
            # 生成化学式
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            formula = generate_chemical_formula(mol)
        
        # 沸点模型路径（使用相对路径，兼容Streamlit Cloud）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'boiling_models')
        
        # 确保模型目录存在
        if not os.path.exists(model_dir):
            return None
        
        predictor = BoilingPointPredictor(model_dir)
        
        if len(predictor.models) == 0:
            return None
        
        result = predictor.predict(formula)
        if result:
            result['smiles'] = smiles
            result['formula'] = formula
            result['chinese_name'] = chinese_name
            result['english_name'] = en
            result['cas'] = cas
        return result
    except Exception as e:
        print(f"predict_boiling_point error: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_solubility(user_input):
    """预测水溶解度"""
    try:
        # 解析用户输入
        result = resolve_molecule_input(user_input)
        if result is None:
            return None
        
        # 检查是否是模糊搜索模式
        if result.get('is_fuzzy_search'):
            user_query = result['english_name'].replace('FUZZY_SEARCH:', '')
            suggestions = result['cas']
            return {'is_fuzzy_search': True, 'user_query': user_query, 'suggestions': suggestions}
        
        smiles = result['smiles']
        en = result['english_name']
        cas = result['cas']
        
        # 水溶解度模型路径（使用相对路径，兼容Streamlit Cloud）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'solubility_models')
        
        # 确保模型目录存在
        if not os.path.exists(model_dir):
            print(f"模型目录不存在: {model_dir}")
            return None
        
        predictor = SolubilityPredictor(model_dir)
        
        if predictor.models is None:
            print("模型未加载")
            return None
        
        pred_result = predictor.predict(smiles)
        if pred_result is not None:
            # SolubilityPredictor.predict 返回的是浮点数，需要包装成字典
            return {
                'smiles': smiles,
                'chinese_name': result.get('chinese_name', ''),
                'english_name': en,
                'cas': cas,
                'formula': result.get('formula', ''),
                'logS': pred_result
            }
        return pred_result
        
    except Exception as e:
        print(f"预测出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_solubility_description(logs, molecule_name=None):
    """根据 LogS 值描述溶解度"""
    # 水的真实参考值（来自实验）
    if molecule_name and molecule_name.lower() == '水':
        return f"⚠️ **注意**: 水的实验 LogS ≈ -1.49 (25°C)。模型预测值可能因训练数据中缺少水分子而存在偏差。"
    
    if logs > 0:
        return "✅ **易溶于水** - 该分子在水中溶解度很高"
    elif logs >= -1:
        return "✅ **可溶于水** - 该分子在水中溶解度良好"
    elif logs >= -2:
        return "⚠️ **微溶于水** - 该分子在水中溶解度较低"
    elif logs >= -3:
        return "⚠️ **极微溶于水** - 该分子几乎不溶于水"
    else:
        return "❌ **几乎不溶于水** - 该分子在水中溶解度极低"


# ==================== 主页面 ====================
def main():
    # 确保模型文件存在
    ensure_model_files()
    
    st.set_page_config(page_title="分子性质预测工具", page_icon="🧪", layout="wide")
    
    # 自定义CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .sub-header {
        color: #2ca02c;
        font-size: 2em;
        margin-bottom: 1em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧪 分子性质预测工具</h1>', unsafe_allow_html=True)
    
    # 侧边栏导航
    page = st.sidebar.radio("选择功能", ["水溶解度预测", "沸点预测"])
    
    if page == "水溶解度预测":
        show_solubility_page()
    else:
        show_boiling_point_page()


# ==================== 启动应用 ====================
if __name__ == "__main__":
    main()
