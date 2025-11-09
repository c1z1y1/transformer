"""
数据加载和预处理模块
支持WikiText-2、PTB、Tiny Shakespeare等数据集
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import zipfile
import re

# 尝试导入requests，如果没有则使用urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False


class TextDataset(Dataset):
    """
    文本数据集类
    """
    
    def __init__(self, texts, vocab, seq_len=128):
        """
        Args:
            texts: 文本列表
            vocab: 词汇表字典
            seq_len: 序列长度
        """
        self.texts = texts
        self.vocab = vocab
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 转换为token索引
        tokens = [self.vocab.get(token, self.vocab['<unk>']) for token in text]
        
        # 截断或填充到固定长度
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens = tokens + [self.vocab['<pad>']] * (self.seq_len - len(tokens))
        
        # 输入和目标（语言模型任务：预测下一个token）
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids


class Vocabulary:
    """
    词汇表类
    """
    
    def __init__(self):
        self.word2idx = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.count = {}
        
    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.count[word] = 1
        else:
            self.count[word] = self.count.get(word, 0) + 1
    
    def build_vocab(self, texts, min_freq=2):
        """
        从文本构建词汇表
        
        Args:
            texts: 文本列表
            min_freq: 最小词频
        """
        # 统计词频
        for text in texts:
            for word in text:
                self.add_word(word)
        
        # 过滤低频词
        if min_freq > 1:
            filtered_words = [word for word, count in self.count.items() 
                            if count >= min_freq or word in ['<pad>', '<unk>', '<bos>', '<eos>']]
            self.word2idx = {word: idx for idx, word in enumerate(filtered_words)}
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
            # 确保特殊token存在
            for token in ['<pad>', '<unk>', '<bos>', '<eos>']:
                if token not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[token] = idx
                    self.idx2word[idx] = token
    
    def __len__(self):
        return len(self.word2idx)
    
    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])


def download_wikitext2(data_dir='./data'):
    """
    下载WikiText-2数据集
    
    Returns:
        解压后的数据集目录路径
    """
    os.makedirs(data_dir, exist_ok=True)
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    zip_path = os.path.join(data_dir, "wikitext-2-v1.zip")
    extracted_dir = os.path.join(data_dir, "wikitext-2")
    
    # 检查解压后的目录和关键文件是否存在
    # 尝试多个可能的路径（因为zip文件可能有不同的内部结构）
    possible_paths = [
        os.path.join(extracted_dir, "wiki.train.tokens"),  # 直接路径
        os.path.join(extracted_dir, "wikitext-2", "wiki.train.tokens"),  # 嵌套路径
    ]
    
    train_file = None
    valid_file = None
    actual_dir = None
    
    # 查找实际存在的文件路径
    for train_path in possible_paths:
        valid_path = train_path.replace("wiki.train.tokens", "wiki.valid.tokens")
        if os.path.exists(train_path) and os.path.exists(valid_path):
            train_file = train_path
            valid_file = valid_path
            actual_dir = os.path.dirname(train_path)
            break
    
    if train_file and valid_file:
        print(f"数据集已存在: {actual_dir}")
        return actual_dir
    
    # 如果zip文件不存在，下载
    if not os.path.exists(zip_path):
        print("正在下载WikiText-2数据集...")
        print(f"URL: {url}")
        try:
            if HAS_REQUESTS:
                # 使用requests库，自动处理重定向
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)
                response.raise_for_status()  # 如果状态码不是200，抛出异常
                
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                with open(zip_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            out_file.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\r下载进度: {percent:.1f}%", end='', flush=True)
                print("\n下载完成")
            else:
                # 回退到urllib，手动处理重定向
                import urllib.request
                import urllib.error
                
                # 创建支持重定向的opener
                opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
                opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
                urllib.request.install_opener(opener)
                
                # 下载文件
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    with open(zip_path, 'wb') as out_file:
                        # 分块下载，避免内存问题
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\r下载进度: {percent:.1f}%", end='', flush=True)
                
                print("\n下载完成")
        except Exception as e:
            print(f"\n下载失败: {e}")
            print("请手动下载数据集:")
            print(f"URL: {url}")
            print(f"保存到: {zip_path}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise
    
    # 如果zip文件存在，解压
    if os.path.exists(zip_path):
        print("正在解压数据集...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 检查zip文件是否有效
                zip_ref.testzip()
                # 解压所有文件
                zip_ref.extractall(data_dir)
            print("解压完成")
        except zipfile.BadZipFile:
            print(f"错误: {zip_path} 不是有效的zip文件")
            print("请删除该文件后重新下载")
            raise
        except Exception as e:
            print(f"解压失败: {e}")
            raise
    
    # 验证解压后的文件是否存在
    # 再次尝试多个可能的路径
    possible_paths = [
        os.path.join(extracted_dir, "wiki.train.tokens"),  # 直接路径
        os.path.join(extracted_dir, "wikitext-2", "wiki.train.tokens"),  # 嵌套路径
    ]
    
    train_file = None
    valid_file = None
    actual_dir = None
    
    # 查找实际存在的文件路径
    for train_path in possible_paths:
        valid_path = train_path.replace("wiki.train.tokens", "wiki.valid.tokens")
        if os.path.exists(train_path) and os.path.exists(valid_path):
            train_file = train_path
            valid_file = valid_path
            actual_dir = os.path.dirname(train_path)
            break
    
    if not train_file or not valid_file:
        # 列出实际存在的文件，帮助调试
        import glob
        all_files = []
        if os.path.exists(extracted_dir):
            for root, dirs, files in os.walk(extracted_dir):
                for f in files:
                    if f.endswith('.tokens'):
                        all_files.append(os.path.join(root, f))
        
        error_msg = (
            f"解压后未找到训练文件或验证文件\n"
            f"预期路径: {os.path.join(extracted_dir, 'wiki.train.tokens')}\n"
            f"或: {os.path.join(extracted_dir, 'wikitext-2', 'wiki.train.tokens')}\n"
        )
        if all_files:
            error_msg += f"实际找到的文件:\n" + "\n".join(f"  - {f}" for f in all_files)
        else:
            error_msg += f"在 {extracted_dir} 中未找到任何 .tokens 文件"
        
        raise FileNotFoundError(error_msg)
    
    print(f"数据集准备完成: {actual_dir}")
    return actual_dir


def load_wikitext2(data_dir='./data', split='train'):
    """
    加载WikiText-2数据集
    
    Args:
        data_dir: 数据目录
        split: 'train', 'valid', 或 'test'
    
    Returns:
        texts: 文本列表
    """
    if split not in ['train', 'valid', 'test']:
        raise ValueError(f"split必须是'train', 'valid'或'test'，当前为: {split}")
    
    data_path = download_wikitext2(data_dir)
    file_path = os.path.join(data_path, f"wiki.{split}.tokens")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"未找到数据文件: {file_path}\n"
            f"请检查数据集是否正确下载和解压"
        )
    
    print(f"正在加载数据文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        raise IOError(f"读取文件失败: {file_path}\n错误: {e}")
    
    # 简单分词（按空格和标点）
    texts = []
    for line in text.split('\n'):
        if line.strip():
            # 简单的tokenization
            tokens = re.findall(r'\b\w+\b|[^\w\s]', line.lower())
            if len(tokens) > 0:
                texts.append(tokens)
    
    print(f"加载完成，共 {len(texts)} 行数据")
    return texts


def load_tiny_shakespeare(data_dir='./data'):
    """
    加载Tiny Shakespeare数据集
    """
    os.makedirs(data_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = os.path.join(data_dir, "tinyshakespeare.txt")
    
    if not os.path.exists(file_path):
        print("正在下载Tiny Shakespeare数据集...")
        try:
            # 创建支持重定向的opener
            opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
            urllib.request.install_opener(opener)
            
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                with open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
            print("下载完成")
        except Exception as e:
            print(f"下载失败: {e}")
            raise
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 字符级tokenization
    texts = []
    for line in text.split('\n'):
        if line.strip():
            tokens = list(line.lower())
            if len(tokens) > 0:
                texts.append(tokens)
    
    return texts


def prepare_data(dataset_name='wikitext2', data_dir='./data', seq_len=128, 
                 batch_size=32, min_freq=2, split_ratio=0.9):
    """
    准备数据加载器
    
    Args:
        dataset_name: 数据集名称 ('wikitext2' 或 'tinyshakespeare')
        data_dir: 数据目录
        seq_len: 序列长度
        batch_size: 批次大小
        min_freq: 最小词频
        split_ratio: 训练集比例
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        vocab: 词汇表
    """
    # 加载数据
    if dataset_name == 'wikitext2':
        train_texts = load_wikitext2(data_dir, 'train')
        val_texts = load_wikitext2(data_dir, 'valid')
    elif dataset_name == 'tinyshakespeare':
        all_texts = load_tiny_shakespeare(data_dir)
        split_idx = int(len(all_texts) * split_ratio)
        train_texts = all_texts[:split_idx]
        val_texts = all_texts[split_idx:]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 构建词汇表
    vocab = Vocabulary()
    vocab.build_vocab(train_texts, min_freq=min_freq)
    print(f"词汇表大小: {len(vocab)}")
    
    # 创建数据集
    train_dataset = TextDataset(train_texts, vocab.word2idx, seq_len)
    val_dataset = TextDataset(val_texts, vocab.word2idx, seq_len)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Windows上设为0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, vocab

