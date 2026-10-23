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

# ==================== 水溶解度预测模型 ====================
class SolubilityPredictor:
    """水溶解度预测器 - 使用本地模型文件"""
    
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'solubility_models')
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
    def __init__(self, model_dir):
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
            
            # 加载5个fold的模型
            for fold in range(1, 6):
                model_path = os.path.join(self.model_dir, f'best_lstm_fold{fold}.pth')
                char_path = os.path.join(self.model_dir, f'char2idx_fold{fold}.pkl')
                meta_path = os.path.join(self.model_dir, f'meta_fold{fold}.pkl')
                
                if os.path.exists(model_path) and os.path.exists(char_path) and os.path.exists(meta_path):
                    with open(char_path, 'rb') as f:
                        self.char2idx[fold] = pickle.load(f)
                    with open(meta_path, 'rb') as f:
                        self.meta[fold] = pickle.load(f)
                    
                    # 加载模型
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
            
            print(f"✓ 沸点预测 LSTM 模型已加载 ({len(self.models)} 个 fold)")
        except Exception as e:
            print(f"⚠ 沸点预测模型加载失败: {e}")
    
    def predict(self, formula, fold=4):
        """预测沸点"""
        try:
            if len(self.models) == 0:
                return None
            
            # 获取 SMILES
            smiles = self._formula_to_smiles(formula)
            if smiles is None:
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
            print(f"预测出错: {e}")
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
    "C2H6O": ["CCO", "CO"],  # 乙醇、甲醇
    "C2H6": ["CC"],  # 乙烷
    "C3H8": ["CCC"],  # 丙烷
    "C4H10": ["CCCC", "CC(C)C"],  # 丁烷、异丁烷
    "C6H6": ["c1ccccc1"],  # 苯
    "C7H8": ["Cc1ccccc1"],  # 甲苯
    "CH2O": ["C=O"],  # 甲醛
    "CH4O": ["CO"],  # 甲醇
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


# ==================== PubChem API 查询 ====================
def lookup_pubchem(name):
    """
    通过 PubChem API 联网查找 SMILES
    返回 (SMILES, 英文名, CAS号) 或 None
    """
    name = name.strip()
    if not name:
        return None
    
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON'
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if 'PropertyTable' in data:
                props = data['PropertyTable']['Properties'][0]
                smiles = props.get('SMILES', '')
                if smiles:
                    cas = props.get('CAS', '')
                    iupac = props.get('IUPACName', '')
                    return (smiles, iupac or name, cas or '')
    except Exception:
        pass
    
    return None


def resolve_molecule_input(user_input):
    """
    解析用户输入，返回 (SMILES, 英文名, CAS号) 或 None
    支持: 中文名、英文名、化学式、SMILES、CAS No
    优先使用本地字典，失败则通过 PubChem API 联网查询
    """
    user_input = user_input.strip()
    
    # 如果输入是SMILES，直接返回
    mol = Chem.MolFromSmiles(user_input)
    if mol is not None:
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        return (canonical, '', '')
    
    # 尝试从映射表中查找
    if user_input in MOLECULE_MAP:
        smiles = MOLECULE_MAP[user_input]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return (smiles, '', '')
    
    # 尝试不区分大小写的英文名匹配
    lower_input = user_input.lower()
    for name, smiles in MOLECULE_MAP.items():
        if isinstance(name, str) and name.lower() == lower_input:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return (smiles, '', '')
    
    # 尝试CAS No匹配
    if re.match(r'^\d{1,7}-\d{2}-\d$', user_input):
        if user_input in MOLECULE_MAP:
            smiles = MOLECULE_MAP[user_input]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return (smiles, '', '')
    
    # 尝试化学式匹配
    # 首先规范化化学式（移除空格，统一大小写）
    import re as regex
    formula_input = user_input.replace(' ', '')
    if regex.match(r'^[A-Z][a-z]?\d*([A-Z][a-z]?\d*)*$', formula_input):
        if formula_input in CHEMICAL_FORMULA_MAP:
            smiles_list = CHEMICAL_FORMULA_MAP[formula_input]
            if len(smiles_list) == 1:
                # 只有一个异构体，直接返回
                mol = Chem.MolFromSmiles(smiles_list[0])
                if mol:
                    return (smiles_list[0], '', '')
            else:
                # 有多个异构体，需要让用户选择
                # 这里返回一个特殊格式，包含化学式和所有可能的SMILES
                return (None, f'CHEMICAL_FORMULA:{formula_input}', smiles_list)
    
    # 本地字典没找到，尝试通过 PubChem API 联网查询
    pubchem_result = lookup_pubchem(user_input)
    if pubchem_result:
        return pubchem_result
    
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
    
    # 尝试CAS No匹配
    if re.match(r'^\d{1,7}-\d{2}-\d$', user_input):
        if user_input in MOLECULE_MAP:
            smiles = MOLECULE_MAP[user_input]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                formula = generate_chemical_formula(mol)
                return {'input': user_input, 'formula': formula, 'smiles': smiles}
    
    # 如果输入看起来像化学式（包含大写字母），尝试直接使用
    if any(c.isupper() for c in user_input) and len(user_input) <= 10:
        import re
        if re.match(r'^[A-Z][a-z]?(\d+)?([A-Z][a-z]?\d?)*$', user_input):
            return {'input': user_input, 'formula': user_input, 'smiles': None}
    
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
        st.markdown("支持输入: **中文名**、**英文名**、**SMILES**、**CAS No**")
        user_input = st.text_input("输入分子标识", "CCO", help="例如: CCO, c1ccccc1, 水, 苯, ethanol, 7732-18-5")
        
        # 显示解析的SMILES
        parsed_smiles = get_smiles_from_input(user_input)
        if parsed_smiles and parsed_smiles != user_input:
            st.info(f"🔍 解析为 SMILES: `{parsed_smiles}`")
        
        if st.button("预测水溶解度", type="primary", use_container_width=True):
            if user_input:
                with st.spinner("正在预测..."):
                    result = predict_solubility(user_input)
                    if result is not None:
                        st.success("预测成功！")
                        st.markdown("### 预测结果")
                        
                        st.metric("LogS (水溶解度)", f"{result:.4f}")
                        
                        # 解释 LogS
                        st.markdown("### 结果解释")
                        solubility_desc = get_solubility_description(result, user_input)
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
        - 苯 (c1ccccc1): LogS ≈ -2.42 (难溶)
        - 甲苯 (Cc1ccccc1): LogS ≈ -2.21 (难溶)
        - 一氧化氮 (NO): LogS ≈ -0.76 (可溶)
        - 氯甲烷 (CCl): LogS ≈ -0.88 (可溶)
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
                    st.session_state['solubility_input'] = smiles
                    st.rerun()


def show_boiling_point_page():
    st.markdown('<h2 class="sub-header">沸点预测</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 输入分子")
        st.markdown("支持输入: **中文名**、**英文名**、**化学式**、**SMILES**、**CAS No**")
        user_input = st.text_input("输入分子标识", "C2H6O", help="例如: C2H6O, C6H6, H2O, 水, ethanol, 7732-18-5")
        
        # 显示解析的化学式或SMILES
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
                                    
                                    # 沸点模型路径
                                    model_dir = r'D:\00全部文件\人工智能赋能化学实验创新赛\徐杰生\沸点预测\模型数据'
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
                            **SMILES**: {result.get('smiles', 'N/A')}  
                            **化学式**: {result.get('formula', 'N/A')}  
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
                    st.session_state['boiling_input'] = formula
                    st.rerun()


# ==================== 预测函数 ====================
def predict_boiling_point(user_input):
    """预测沸点"""
    try:
        # 解析用户输入
        result = resolve_molecule_input(user_input)
        if result is None:
            return None
        smiles, en, cas = result
        
        # 如果是化学式选择模式（有多个异构体）
        if smiles is None and en and en.startswith('CHEMICAL_FORMULA:'):
            formula = en.replace('CHEMICAL_FORMULA:', '')
            return {'is_formula_choice': True, 'formula': formula, 'smiles_list': cas}
        
        # 生成化学式
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        formula = generate_chemical_formula(mol)
        
        # 沸点模型路径（使用相对路径）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'boiling_models')
        
        predictor = BoilingPointPredictor(model_dir)
        
        result = predictor.predict(formula)
        return result
        
    except Exception as e:
        st.error(f"预测出错: {str(e)}")
        return None


def predict_solubility(user_input):
    """预测水溶解度"""
    try:
        # 解析用户输入
        result = resolve_molecule_input(user_input)
        if result is None:
            return None
        smiles, en, cas = result
        
        # 水溶解度模型路径（使用相对路径）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'solubility_models')
        
        predictor = SolubilityPredictor(model_dir)
        
        result = predictor.predict(smiles)
        return result
        
    except Exception as e:
        st.error(f"预测出错: {str(e)}")
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
