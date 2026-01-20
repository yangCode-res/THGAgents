import re
from pathlib import Path

from markdown_it import MarkdownIt

REF_TITLES = {"references", "参考文献"}

def get_mixed_word_count(text: str) -> int:
    """
    计算混合文本的“字数”：
    1. 中文（CJK统一表意文字）：按字符数计算。
    2. 英文/数字/其他：按单词数（空格分隔）计算。
    """
    if not text:
        return 0
        
    # 匹配中文字符的正则范围
    cjk_pattern = re.compile(r'[\u4e00-\u9fff]')
    
    # 1. 统计中文字符数量
    cjk_chars = cjk_pattern.findall(text)
    cjk_count = len(cjk_chars)
    
    # 2. 将中文字符替换为空格，剩余的按空格切分统计英文单词
    text_without_cjk = cjk_pattern.sub(' ', text)
    english_word_count = len(text_without_cjk.split())
    
    return cjk_count + english_word_count

def remove_references(text: str) -> str:
    """
    在内存中移除 Markdown 文本中从第一个 `## References` / `## 参考文献` 开始的内容。
    """
    md = MarkdownIt()
    tokens = md.parse(text)

    cut_line = None
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag == "h2" and tok.map:
            if i + 1 < len(tokens) and tokens[i+1].type == "inline":
                title = tokens[i + 1].content.strip().lower()
                title = re.sub(r"[:：]", "", title)
                
                if title in REF_TITLES or any(title.startswith(t) for t in REF_TITLES):
                    cut_line = tok.map[0]
                    break
    
    if cut_line is not None:
        lines = text.splitlines(keepends=True)
        return "".join(lines[:cut_line])
    
    return text

def split_md_by_mixed_count(path: str, target_count: int = 800, min_count: int = 200) -> dict:
    """
    按照“英文算词、中文算字”的方式切分 Markdown。
    
    Args:
        path: 文件路径
        target_count: 目标字数（切块阈值，默认1200）
        min_count: 最小字数限制（少于此值的块会被直接丢弃，默认500）
        
    Returns:
        {filename: [chunk1, chunk2, ...]}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found.")

    raw_text = p.read_text(encoding="utf-8", errors="ignore")
    
    # 1. 清洗参考文献
    clean_text = remove_references(raw_text)

    # 2. 按自然段落拆分 (以空行为界)
    lines = clean_text.splitlines()
    paragraphs = []
    buf = []
    for line in lines:
        if line.strip() == "":
            if buf:
                paragraphs.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        paragraphs.append("\n".join(buf))

    # 3. 贪婪合并切块（生成候选块）
    raw_chunks = []
    current_chunk_paras = []
    current_count = 0

    for para in paragraphs:
        para_count = get_mixed_word_count(para)
        
        # 如果当前积累量 + 新段落 > 目标值，则切分
        if current_chunk_paras and (current_count + para_count > target_count):
            raw_chunks.append("\n\n".join(current_chunk_paras))
            current_chunk_paras = [para]
            current_count = para_count
        else:
            current_chunk_paras.append(para)
            current_count += para_count

    # 处理最后一个候选块
    if current_chunk_paras:
        raw_chunks.append("\n\n".join(current_chunk_paras))

    # 4. [新功能] 过滤掉字数不足的块
    valid_chunks = []
    for chunk in raw_chunks:
        # 重新计算最终块的字数（以确保准确）
        if get_mixed_word_count(chunk) >= min_count:
            valid_chunks.append(chunk)

    return {p.name.replace(".md", ""): valid_chunks}

