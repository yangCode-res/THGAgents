import base64
import os
import pathlib
import re  # 引入正则模块

from mistralai import Mistral

API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL = "mistral-ocr-latest"

def encode_pdf_base64(path):
    with open(path, "rb") as f:
        return "data:application/pdf;base64," + base64.b64encode(f.read()).decode()

def remove_references(text):
    """
    利用正则查找 Markdown 标题中的 References/Bibliography/参考文献，
    并截断之后的内容。
    """
    if not text:
        return text

    # 正则逻辑解释：
    # (?i)       : 开启忽略大小写模式
    # ^          : 匹配行首 (配合 re.MULTILINE)
    # \#+        : 匹配一个或多个 # (Markdown 标题)
    # \s+        : 匹配标题后的空格
    # (\d+\.?\s*)? : 可选匹配章节号 (例如 "10. References" 或 "6 References")
    # (References|Bibliography|参考文献) : 匹配核心关键词
    # \s*        : 匹配尾部可能存在的空格
    # $          : 匹配行尾
    pattern = re.compile(r'(?i)^#+\s+(\d+\.?\s*)?(References|Bibliography|参考文献)\s*$', re.MULTILINE)

    # 搜索匹配项
    match = pattern.search(text)
    
    if match:
        print(f"   -> Detected References section at index {match.start()}, truncating...")
        # 返回匹配位置之前的所有文本，并去除尾部空白
        return text[:match.start()].strip()
    
    return text

def ocr_from_urls(url_list):
    """返回每个 URL 的 OCR 文本 (已去除参考文献)"""
    results = []

    with Mistral(api_key=API_KEY) as client:
        for url in url_list:
            print("Processing:", url)

            try:
                # 判断 URL vs 本地路径
                if url.startswith("http://") or url.startswith("https://"):
                    document_payload = {
                        "document_url": url,
                        "type": "document_url"
                    }
                else:
                    b64 = encode_pdf_base64(url)
                    document_payload = {
                        "document_base64": b64,
                        "type": "document_base64"
                    }

                res = client.ocr.process(
                    model=MODEL,
                    document=document_payload
                )

                # 合并页内容
                pages = []
                for p in res.pages:
                    if getattr(p, "markdown", None):
                        pages.append(p.markdown)
                    elif getattr(p, "text", None):
                        pages.append(p.text)
                
                # 1. 先合并所有页面文本
                full_text = "\n\n".join(pages)
                
                # 2. 执行去除参考文献的逻辑
                cleaned_text = remove_references(full_text)
                
                results.append(cleaned_text)

            except Exception as e:
                print("Error:", e)
                results.append(None)

    return results


# ----------------------------------------------------
# 📌 包装函数：输入 URL 列表 → 输出保存的 MD 文件路径列表
# ----------------------------------------------------
def ocr_to_md_files(url_list, save_dir="ocr_md_outputs", start_index: int = 1):
    """
    输入: url_list = [url1, url2, ...]
    输出: md_paths = ["xxx/file1.md", "xxx/file2.md", ...]

    start_index: 用于避免多次调用时文件名被覆盖，例如 start_index=3 会生成 ocr_result_3.md 起步。
    """
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    md_paths = []

    texts = ocr_from_urls(url_list)

    for idx, text in enumerate(texts, start=start_index):
        if text is None:
            md_paths.append(None)
            continue

        md_path = save_dir / f"ocr_result_{idx}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)

        md_paths.append(str(md_path))

    return md_paths


