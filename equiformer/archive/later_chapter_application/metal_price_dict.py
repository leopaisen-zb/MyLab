#!/usr/bin/env python3
"""
金属价格字典生成器 - 基于USGS MCS数据

该模块从美国地质调查局（USGS）矿物商品概要（MCS）获取权威金属价格数据，
并统一转换为美元/千克的标准单位。

数据来源优先级：
1. USGS MCS 2025 ZIP文件 - 从CSV中提取price_*列
2. USGS MCS 2025 PDF - 使用tabula-py解析"Price, annual average"数据  
3. USGS MCS 2024 PDF - 备用PDF数据源
4. 内置USGS历史均价 - 最终后备数据

所有价格最终统一为：USD/kg (美元/千克)

Features:
- 自动下载和解析USGS官方数据
- 智能单位识别和转换
- 多级降级策略确保数据可用性
- 详细的处理日志和错误处理

Author: AI Assistant  
Date: 2025-01-XX
Version: 2.0
"""

import json
import requests
import pandas as pd
import zipfile
import io
import re
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path

# 尝试导入tabula-py，如果没有安装则给出提示
try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    print("[WARN] tabula-py未安装，PDF解析功能将不可用。安装命令: pip install tabula-py")

# ==================== 常数定义 ====================
# 单位转换常数
OZ_TO_KG = 0.0311034768  # 1 troy ounce = 0.0311034768 kg
TONNES_TO_KG = 1000      # 1 tonne = 1000 kg
GRAMS_TO_KG = 0.001      # 1 gram = 0.001 kg
LB_TO_KG = 0.453592      # 1 pound = 0.453592 kg
GRAM_PER_TONNE = 1000000 # 1 tonne = 1,000,000 grams

# 网络请求超时设置
REQUEST_TIMEOUT = 30  # 秒，PDF文件可能较大

# ==================== USGS MCS 数据源配置 ====================
# USGS MCS 2025 数据源
USGS_MCS_2025_ZIP_URL = "https://pubs.usgs.gov/periodicals/mcs2025/data/Salient_Commodity_Data_Release_Grouped_MCS_2025.zip"
USGS_MCS_2025_PDF_URL = "https://pubs.usgs.gov/periodicals/mcs2025/mcs2025.pdf"

# USGS MCS 2024 备份数据源
USGS_MCS_2024_PDF_URL = "https://pubs.usgs.gov/periodicals/mcs2024/mcs2024.pdf"

# 临时文件目录
TEMP_DIR = Path("./temp_usgs_data")

# 元素符号到商品名称的映射（用于PDF解析）
ELEMENT_TO_COMMODITY = {
    "Au": ["Gold", "GOLD"],
    "Ag": ["Silver", "SILVER"], 
    "Pt": ["Platinum", "PLATINUM"],
    "Pd": ["Palladium", "PALLADIUM"],
    "Cu": ["Copper", "COPPER"],
    "Ni": ["Nickel", "NICKEL"],
    "Al": ["Aluminum", "ALUMINUM", "Aluminium", "ALUMINIUM"],
    "Zn": ["Zinc", "ZINC"],
    "Pb": ["Lead", "LEAD"],
    "Sn": ["Tin", "TIN"],
    "Fe": ["Iron ore", "IRON ORE", "Iron", "IRON"],
    "Co": ["Cobalt", "COBALT"],
    "Mo": ["Molybdenum", "MOLYBDENUM"],
    "W": ["Tungsten", "TUNGSTEN"],
    "Cr": ["Chromium", "CHROMIUM"],
    "Mn": ["Manganese", "MANGANESE"],
    "V": ["Vanadium", "VANADIUM"],
    "Ti": ["Titanium", "TITANIUM"],
    "Ta": ["Tantalum", "TANTALUM"],
    "Nb": ["Niobium", "NIOBIUM"],
    "Zr": ["Zirconium", "ZIRCONIUM"],
    "Li": ["Lithium", "LITHIUM"],
    "Be": ["Beryllium", "BERYLLIUM"],
    "Mg": ["Magnesium", "MAGNESIUM"]
}

# USGS历史均价后备数据 (USD/kg) - 基于MCS 2024数据
USGS_HISTORICAL_FALLBACK = {
    "Au": 65000,    # 金 (~$2000/oz)
    "Ag": 24000,    # 银 (~$24/oz)
    "Pt": 30500,    # 铂 (~$950/oz)
    "Pd": 7000,     # 钯 (~$220/oz)
    "Cu": 8.5,      # 铜 (~$8500/tonne)
    "Ni": 24,       # 镍 (~$24000/tonne)
    "Al": 1.8,      # 铝 (~$1800/tonne)
    "Zn": 2.8,      # 锌 (~$2800/tonne)
    "Pb": 2.1,      # 铅 (~$2100/tonne)
    "Sn": 25,       # 锡 (~$25000/tonne)
    "Fe": 0.12,     # 铁矿石 (~$120/tonne)
    "Co": 33,       # 钴 (~$33/kg)
    "Mo": 30,       # 钼 (~$30/kg)
    "W": 35,        # 钨 (~$35/kg)
    "Cr": 0.3,      # 铬铁 (~$300/tonne)
    "Mn": 2,        # 锰 (~$2000/tonne)
    "V": 27,        # 钒 (~$27/kg)
    "Ti": 4.5,      # 钛 (~$4.5/kg)
    "Ta": 210,      # 钽 (~$210/kg)
    "Nb": 40,       # 铌 (~$40/kg)
    "Zr": 160,      # 锆 (~$160/kg)
    "Li": 15,       # 锂 (~$15/kg)
    "Be": 320,      # 铍 (~$320/kg)
    "Mg": 3.2       # 镁 (~$3200/tonne)
}


def ensure_temp_dir() -> None:
    """
    确保临时目录存在
    """
    TEMP_DIR.mkdir(exist_ok=True)


def clean_temp_dir() -> None:
    """
    清理临时目录
    """
    try:
        if TEMP_DIR.exists():
            for file in TEMP_DIR.glob("*"):
                file.unlink()
            TEMP_DIR.rmdir()
    except Exception as e:
        print(f"[WARN] 清理临时目录失败: {e}")


def parse_price_value(price_str: str) -> Optional[float]:
    """
    解析价格字符串，提取数值和单位
    
    Args:
        price_str (str): 价格字符串，如 "$8,500/tonne", "24.5 $/oz"
        
    Returns:
        Optional[float]: 解析出的价格值，失败时返回None
    """
    if not price_str or pd.isna(price_str):
        return None
    
    # 清理字符串
    clean_str = str(price_str).strip().replace(",", "").replace("$", "")
    
    # 尝试提取数字
    numbers = re.findall(r"[\d,]+\.?\d*", clean_str)
    if not numbers:
        return None
    
    try:
        return float(numbers[0].replace(",", ""))
    except ValueError:
        return None


def convert_to_usd_per_kg(price: float, unit_str: str) -> Optional[float]:
    """
    将价格转换为USD/kg
    
    Args:
        price (float): 价格数值
        unit_str (str): 单位字符串
        
    Returns:
        Optional[float]: 转换后的USD/kg价格
    """
    unit_lower = unit_str.lower()
    
    # Troy ounce相关单位
    if any(keyword in unit_lower for keyword in ["oz", "ounce", "troy"]):
        return price / OZ_TO_KG
    
    # 吨相关单位
    elif any(keyword in unit_lower for keyword in ["tonne", "ton", "mt", "metric"]):
        return price / TONNES_TO_KG
        
    # 磅相关单位
    elif any(keyword in unit_lower for keyword in ["lb", "pound"]):
        return price / LB_TO_KG
        
    # 克相关单位
    elif any(keyword in unit_lower for keyword in ["gram", "g/"]):
        return price / GRAMS_TO_KG
        
    # 千克相关单位（已经是目标单位）
    elif any(keyword in unit_lower for keyword in ["kg", "kilogram"]):
        return price
        
    # 默认假设为USD/kg
    else:
        print(f"[WARN] 未识别的单位 '{unit_str}'，假设为USD/kg")
        return price


def download_usgs_zip() -> Optional[Path]:
    """
    下载USGS MCS 2025 ZIP文件
    
    Returns:
        Optional[Path]: 下载的文件路径，失败时返回None
    """
    try:
        print("[INFO] 正在下载USGS MCS 2025 ZIP文件...")
        response = requests.get(USGS_MCS_2025_ZIP_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        zip_path = TEMP_DIR / "mcs2025.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
            
        print(f"[INFO] ZIP文件下载成功: {zip_path}")
        return zip_path
        
    except Exception as e:
        print(f"[WARN] USGS ZIP下载失败: {e}")
        return None


def extract_prices_from_zip(zip_path: Path) -> Dict[str, float]:
    """
    从ZIP文件中的CSV提取价格数据
    
    Args:
        zip_path (Path): ZIP文件路径
        
    Returns:
        Dict[str, float]: {元素符号: 价格USD/kg}的字典
    """
    prices = {}
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 列出ZIP文件中的所有文件
            file_list = zip_ref.namelist()
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            print(f"[INFO] 在ZIP中找到 {len(csv_files)} 个CSV文件")
            
            for csv_file in csv_files:
                try:
                    # 读取CSV文件
                    with zip_ref.open(csv_file) as csv_data:
                        df = pd.read_csv(csv_data)
                        
                    # 查找price相关列
                    price_columns = [col for col in df.columns if 'price' in col.lower()]
                    
                    if not price_columns:
                        continue
                        
                    print(f"[INFO] 在 {csv_file} 中找到价格列: {price_columns}")
                    
                    # 尝试从商品名识别元素
                    commodity_info = extract_commodity_info_from_filename(csv_file)
                    if not commodity_info:
                        continue
                        
                    element, commodity_name = commodity_info
                    
                    # 提取价格数据
                    for price_col in price_columns:
                        # 获取最新的非空价格值
                        price_series = df[price_col].dropna()
                        if len(price_series) > 0:
                            latest_price = price_series.iloc[-1]
                            
                            # 解析价格值
                            price_value = parse_price_value(str(latest_price))
                            if price_value and price_value > 0:
                                # 尝试从列名推断单位
                                unit_info = extract_unit_from_column_name(price_col)
                                if unit_info:
                                    price_kg = convert_to_usd_per_kg(price_value, unit_info)
                                    if price_kg:
                                        prices[element] = round(price_kg, 2)
                                        print(f"  {element} ({commodity_name}): ${price_kg:.2f}/kg")
                                        break
                        
                except Exception as e:
                    print(f"[WARN] 解析CSV文件 {csv_file} 失败: {e}")
                    continue
                    
    except Exception as e:
        print(f"[WARN] ZIP文件解析失败: {e}")
        
    return prices


def extract_commodity_info_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    从文件名提取商品信息
    
    Args:
        filename (str): CSV文件名
        
    Returns:
        Optional[Tuple[str, str]]: (元素符号, 商品名)，失败时返回None
    """
    filename_lower = filename.lower()
    
    for element, commodity_names in ELEMENT_TO_COMMODITY.items():
        for commodity in commodity_names:
            if commodity.lower() in filename_lower:
                return element, commodity
                
    return None


def extract_unit_from_column_name(column_name: str) -> Optional[str]:
    """
    从列名提取单位信息
    
    Args:
        column_name (str): 列名
        
    Returns:
        Optional[str]: 单位字符串
    """
    col_lower = column_name.lower()
    
    # 常见单位模式
    unit_patterns = [
        r"per\s+(\w+)",
        r"/(\w+)",
        r"\(([^)]+)\)",
        r"_(\w+)$"
    ]
    
    for pattern in unit_patterns:
        match = re.search(pattern, col_lower)
        if match:
            return match.group(1)
            
    return None


def download_usgs_pdf(url: str, filename: str) -> Optional[Path]:
    """
    下载USGS PDF文件
    
    Args:
        url (str): PDF文件URL
        filename (str): 本地文件名
        
    Returns:
        Optional[Path]: 下载的文件路径，失败时返回None
    """
    try:
        print(f"[INFO] 正在下载USGS PDF: {filename}...")
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        pdf_path = TEMP_DIR / filename
        with open(pdf_path, "wb") as f:
            f.write(response.content)
            
        print(f"[INFO] PDF文件下载成功: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        print(f"[WARN] PDF下载失败 ({filename}): {e}")
        return None


def extract_prices_from_pdf(pdf_path: Path) -> Dict[str, float]:
    """
    使用tabula-py从PDF提取价格数据
    
    Args:
        pdf_path (Path): PDF文件路径
        
    Returns:
        Dict[str, float]: {元素符号: 价格USD/kg}的字典
    """
    if not TABULA_AVAILABLE:
        print("[WARN] tabula-py不可用，跳过PDF解析")
        return {}
        
    prices = {}
    
    try:
        print(f"[INFO] 正在解析PDF文件: {pdf_path.name}")
        
        # 使用更安全的参数配置
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # 尝试读取PDF，使用更宽松的配置
        try:
            # 先尝试只读取前几页，避免图像处理问题
            tables = tabula.read_pdf(
                str(pdf_path), 
                pages="1-5",  # 只读取前5页
                multiple_tables=True,
                silent=True,  # 静默模式，减少Java错误输出
                pandas_options={'header': None}  # 不假设有标题行
            )
        except Exception as e:
            print(f"[WARN] 尝试前5页解析失败，改为尝试第1页: {e}")
            try:
                tables = tabula.read_pdf(
                    str(pdf_path), 
                    pages=1,  # 只读取第1页
                    multiple_tables=True,
                    silent=True
                )
            except Exception as e2:
                print(f"[WARN] PDF表格提取完全失败: {e2}")
                return {}
        
        if not tables:
            print("[WARN] PDF中未找到任何表格")
            return {}
            
        print(f"[INFO] 在PDF中成功提取到 {len(tables)} 个表格")
        
        # 解析提取到的表格
        for i, table in enumerate(tables):
            try:
                if table.empty:
                    continue
                    
                print(f"[INFO] 处理表格 {i+1}，尺寸: {table.shape}")
                
                # 查找包含价格相关信息的行
                price_rows = []
                
                for idx, row in table.iterrows():
                    try:
                        # 将行转换为字符串进行搜索
                        row_str = " ".join([str(cell).lower() for cell in row if pd.notna(cell)])
                        
                        # 查找包含价格关键词的行
                        if any(keyword in row_str for keyword in ["price", "annual", "average", "$"]):
                            price_rows.append((idx, row))
                            
                    except Exception as e:
                        continue
                
                if not price_rows:
                    print(f"[INFO] 表格 {i+1} 中未找到价格相关行")
                    continue
                    
                print(f"[INFO] 在表格 {i+1} 中找到 {len(price_rows)} 个潜在价格行")
                
                # 尝试解析找到的价格行
                for idx, row in price_rows:
                    try:
                        # 获取第一列作为商品名
                        commodity_name = str(row.iloc[0]) if len(row) > 0 and pd.notna(row.iloc[0]) else ""
                        
                        # 匹配元素符号
                        element = None
                        for elem, names in ELEMENT_TO_COMMODITY.items():
                            if any(name.lower() in commodity_name.lower() for name in names):
                                element = elem
                                break
                        
                        if not element:
                            continue
                        
                        # 在行中查找价格值
                        for j, cell in enumerate(row[1:], 1):  # 跳过第一列
                            if pd.notna(cell):
                                price_str = str(cell)
                                price_value = parse_price_value(price_str)
                                
                                if price_value and price_value > 0:
                                    # 根据商品类型推断单位并转换
                                    if element in ["Au", "Ag", "Pt", "Pd"]:  # 贵金属通常用盎司
                                        price_kg = convert_to_usd_per_kg(price_value, "oz")
                                    else:  # 其他金属通常用吨
                                        price_kg = convert_to_usd_per_kg(price_value, "tonne")
                                    
                                    if price_kg and price_kg > 0:
                                        prices[element] = round(price_kg, 2)
                                        print(f"  ✅ {element} ({commodity_name.strip()}): ${price_kg:.2f}/kg")
                                        break
                                        
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"[WARN] 处理表格 {i+1} 时出错: {e}")
                continue
                
    except Exception as e:
        print(f"[WARN] PDF解析过程中出现错误: {e}")
        
    if prices:
        print(f"[INFO] PDF解析成功提取到 {len(prices)} 种金属价格")
    else:
        print("[WARN] PDF解析未能提取到任何金属价格")
        
    return prices


def extract_prices_from_pdf_simple(pdf_path: Path) -> Dict[str, float]:
    """
    使用PyPDF2作为备用方案从PDF提取价格数据
    
    Args:
        pdf_path (Path): PDF文件路径
        
    Returns:
        Dict[str, float]: {元素符号: 价格USD/kg}的字典
    """
    try:
        import PyPDF2
        print(f"[INFO] 使用PyPDF2备用方案解析PDF: {pdf_path.name}")
        
        prices = {}
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # 只读取前5页，避免处理过多内容
            max_pages = min(5, len(pdf_reader.pages))
            
            for page_num in range(max_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # 按行分割文本
                    lines = text.split('\n')
                    
                    for line in lines:
                        line_lower = line.lower()
                        
                        # 查找包含价格关键词的行
                        if 'price' in line_lower and ('annual' in line_lower or 'average' in line_lower):
                            # 尝试匹配元素
                            for element, commodity_names in ELEMENT_TO_COMMODITY.items():
                                for commodity in commodity_names:
                                    if commodity.lower() in line_lower:
                                        # 提取价格
                                        price_match = re.search(r'\$[\d,]+\.?\d*', line)
                                        if price_match:
                                            price_str = price_match.group()
                                            price_value = parse_price_value(price_str)
                                            
                                            if price_value and price_value > 0:
                                                # 根据商品类型推断单位
                                                if element in ["Au", "Ag", "Pt", "Pd"]:
                                                    price_kg = convert_to_usd_per_kg(price_value, "oz")
                                                else:
                                                    price_kg = convert_to_usd_per_kg(price_value, "tonne")
                                                
                                                if price_kg and price_kg > 0:
                                                    prices[element] = round(price_kg, 2)
                                                    print(f"  ✅ {element} ({commodity}): ${price_kg:.2f}/kg (PyPDF2)")
                                                    break
                                        break
                                
                except Exception as e:
                    continue
                    
        return prices
        
    except ImportError:
        print("[WARN] PyPDF2未安装，无法使用备用PDF解析")
        return {}
    except Exception as e:
        print(f"[WARN] PyPDF2 PDF解析失败: {e}")
        return {}


def fetch_usgs_mcs_data() -> Dict[str, float]:
    """
    获取USGS MCS金属价格数据
    
    按优先级尝试多个数据源：
    1. MCS 2025 ZIP文件
    2. MCS 2025 PDF文件  
    3. MCS 2024 PDF文件
    4. 历史后备数据
    
    Returns:
        Dict[str, float]: {元素符号: 价格USD/kg}的字典
    """
    print("=" * 60)
    print("开始获取USGS MCS金属价格数据")
    print("=" * 60)
    
    ensure_temp_dir()
    prices = {}
    
    try:
        # 第一级：尝试MCS 2025 ZIP文件
        print("\n1️⃣  尝试USGS MCS 2025 ZIP文件:")
        zip_path = download_usgs_zip()
        if zip_path:
            zip_prices = extract_prices_from_zip(zip_path)
            prices.update(zip_prices)
            
        # 第二级：如果数据不足，尝试MCS 2025 PDF
        if len(prices) < 5:
            print("\n2️⃣  尝试USGS MCS 2025 PDF:")
            pdf_path = download_usgs_pdf(USGS_MCS_2025_PDF_URL, "mcs2025.pdf")
            if pdf_path:
                # 首先尝试tabula-py
                pdf_prices = extract_prices_from_pdf(pdf_path)
                
                # 如果tabula-py没有提取到数据，尝试PyPDF2备用方案
                if not pdf_prices:
                    print("[INFO] tabula-py未能提取数据，尝试PyPDF2备用方案...")
                    pdf_prices = extract_prices_from_pdf_simple(pdf_path)
                
                for element, price in pdf_prices.items():
                    if element not in prices:
                        prices[element] = price
        
        # 第三级：如果仍然不足，尝试MCS 2024 PDF备份
        if len(prices) < 3:
            print("\n3️⃣  尝试USGS MCS 2024 PDF备份:")
            pdf_path = download_usgs_pdf(USGS_MCS_2024_PDF_URL, "mcs2024.pdf")
            if pdf_path:
                # 同样使用双重解析策略
                pdf_prices = extract_prices_from_pdf(pdf_path)
                
                if not pdf_prices:
                    print("[INFO] tabula-py未能提取数据，尝试PyPDF2备用方案...")
                    pdf_prices = extract_prices_from_pdf_simple(pdf_path)
                
                for element, price in pdf_prices.items():
                    if element not in prices:
                        prices[element] = price
        
        # 第四级：使用历史后备数据填补缺失
        print("\n4️⃣  使用USGS历史数据填补缺失:")
        missing_count = 0
        for element, fallback_price in USGS_HISTORICAL_FALLBACK.items():
            if element not in prices:
                prices[element] = fallback_price
                print(f"  {element}: ${fallback_price:.2f}/kg (来源: 历史数据)")
                missing_count += 1
        
        if missing_count == 0:
            print("  ✅ 所有金属价格已从USGS官方数据获取")
            
    finally:
        # 清理临时文件
        clean_temp_dir()
    
    print(f"\n🎯 USGS MCS数据获取完成！共包含 {len(prices)} 种金属")
    return prices


def build_price_dict() -> Dict[str, float]:
    """
    构建完整的金属价格字典
    
    主要使用USGS MCS数据作为权威价格来源
    
    Returns:
        Dict[str, float]: {元素符号: 价格_USD/kg}的完整字典
    """
    return fetch_usgs_mcs_data()


def main() -> None:
    """
    主函数：生成金属价格字典并输出JSON格式
    """
    # 生成时间戳
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"# Metal price dictionary (USD/kg) — generated {timestamp}")
    print("")
    
    # 构建价格字典
    try:
        price_dict = build_price_dict()
        
        # 输出JSON格式
        print("\n" + "=" * 60)
        print("📊 最终金属价格字典 (JSON格式):")
        print("=" * 60)
        print(json.dumps(price_dict, indent=4, sort_keys=True))
        
    except Exception as e:
        print(f"❌ 错误：金属价格字典生成失败: {e}")
        # 输出完整后备字典作为最后保障
        print("\n🔄 使用完整USGS历史后备价格字典:")
        print(json.dumps(USGS_HISTORICAL_FALLBACK, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()