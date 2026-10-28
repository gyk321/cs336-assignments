import regex as re
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# 将目标路径的文本文件以 UTF-8 编码完整读入内存
def read_text(input_path):
    with open(input_path,"r",encoding = "utf-8") as f:
        text = f.read()
    return text

# 使用正则表达式，根据指定的特殊 Token（如 <|endoftext|>）将长文本切分成多个独立的文本块
def split_by_special(text,special_tokens,drop_special=True):
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens,key = len,reverse = True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special:
        pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]

# 将单个字符串转化为单字节 bytes 对象组成的元组
def word2bytes(word):
    a = list(word.encode('utf-8'))
    return tuple(bytes([i])  for i in a)

def count_word(text):
    word_cnt = defaultdict(int) 
    for m in PAT.finditer(text):        # 对传入的文本块进行预分词
        word = m.group(0)
        word_bytes = word2bytes(word)   # 将每个切出来的词通过 word2bytes 转为字节元组
        if len(word_bytes) >= 2:
            word_cnt[word_bytes] += 1
    return word_cnt


def merge_dicts(dicts):
    merged = defaultdict(int)
    for d in dicts:
        for k,v in d.items():
            merged[k] += v
    return merged

def count_pair(word_cnt):
    pair_cnt = defaultdict(int)
    for word_bytes,cnt in word_cnt.items():
        for pair in zip(word_bytes[:-1],word_bytes[1:]):
            pair_cnt[pair] += cnt
    return pair_cnt

def get_max_pair(pair_cnt):
    max_pair,_ = max(pair_cnt.items(),key = lambda x:(x[1],x[0]))
    return max_pair

def get_basic_vocab(special_tokens):
    vocab = {token:bytes([token]) for token in range(256)}

    for i,token in enumerate(special_tokens):
        token_id = 256 + i
        vocab[token_id] = token.encode("utf-8")
    return vocab

def apply_merge(word_bytes,merge):
    merged = merge[0] + merge[1]
    i = 0
    new_word_bytes = []
    while i < len(word_bytes):
        if i < len(word_bytes) - 1 and word_bytes[i] == merge[0] and word_bytes[i + 1] == merge[1]:
            new_word_bytes.append(merged)
            i += 2
        else:
            new_word_bytes.append(word_bytes[i])
            i += 1
    return tuple(new_word_bytes)

def update_cnt(word_cnt,pair_cnt,merge_pair):
    new_word_cnt = defaultdict(int)
    new_pair_cnt = defaultdict(int,pair_cnt)

    for word_bytes,cnt in word_cnt.items():
        old_pairs = list(zip(word_bytes[:-1],word_bytes[1:]))

        if merge_pair not in old_pairs:
            new_word_cnt[word_bytes] += cnt
            continue

        new_word = apply_merge(word_bytes,merge_pair)
        new_word_cnt[new_word] += cnt

        for pair in old_pairs:
            new_pair_cnt[pair] -= cnt
            if new_pair_cnt[pair] == 0:
                del new_pair_cnt[pair]

        new_pairs = list(zip(new_word[:-1],new_word[1:]))
        for p in new_pairs:
            new_pair_cnt[p] += cnt
    return new_word_cnt,new_pair_cnt




def train_bpe(input_path,vocab_size,special_tokens):
    text = read_text(input_path)                        # 从指定路径读取完整的训练文本内容
    chunks = split_by_special(text,special_tokens)      # 将一整块大文本切分成多个独立的文本块

    if len(chunks) < 4 :    # 文本块较少
        word_dicts = list(map(count_word,chunks))       # 单线程 map 函数逐块处理
    else:
        word_dicts = process_map(count_word,chunks,chunksize = 1)   # 调用 process_map 开启多进程并行处理

    word_cnt = merge_dicts(word_dicts)                  # 合并成一个全局的词频字典
    pair_cnt = count_pair(word_cnt)                     # 统计出所有相邻字节对（Pair）的初始出现频次

    vocab = get_basic_vocab(special_tokens)             # 初始化基础词表
    base_vocab_size = len(vocab)
    n_merges = vocab_size - base_vocab_size             # 计算迭代次数

    merges = []

    for i  in range(n_merges):
        max_pair = get_max_pair(pair_cnt)               # 从当前的 pair_cnt 中找出出现次数最多的那个字节对
        vocab[base_vocab_size + i] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        word_cnt,pair_cnt = update_cnt(word_cnt,pair_cnt,max_pair)

    return vocab,merges