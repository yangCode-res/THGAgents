

import hashlib
import os
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests


# --------- Helpers for safe filename ----------
def _clean_name(s: str, maxlen: int = 200) -> str:
    # 保留字母数字和这些字符，去掉其它危险字符
    safe = "".join(c for c in s if c.isalnum() or c in " .-_()[]")
    safe = safe.strip()
    if len(safe) > maxlen:
        safe = safe[:maxlen]
    return safe or None

def make_safe_filename_from_url(url: Optional[str], fallback_prefix: str = "file", maxlen: int = 200) -> Optional[str]:

    if not url:
        return None

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    qs = parse_qs(parsed.query or "")
    candidates = []
    for key in ("accid", "id", "pmcid", "file", "filename"):
        v = qs.get(key)
        if v:
            candidates.append(v[0])

    for cand in candidates:
        if cand:
            name = unquote(str(cand))
            # remove weird chars
            name = re.sub(r"[^\w\-\.\(\)\[\] ]+", "", name)
            c = _clean_name(name, maxlen=maxlen-4)
            if c:
                if not c.lower().endswith(".pdf"):
                    c = c + ".pdf"
                return c

    # try basename of path
    path = unquote(parsed.path or "")
    base = os.path.basename(path)
    if base:
        base = base.split("?")[0].split("#")[0]
        base_clean = _clean_name(base, maxlen=maxlen)
        if base_clean:
            if base.lower().endswith(".pdf"):
                return base_clean if base_clean.lower().endswith(".pdf") else base_clean + ".pdf"
            else:
                return base_clean + ".pdf"

    # fallback: hash of url
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    name = f"{fallback_prefix}_{h}.pdf"
    return _clean_name(name, maxlen=maxlen)

# --------- Minimal PDF download ----------
def download_pdf(url: str, save_path: str, timeout: int = 20) -> bool:

    if not url or not str(url).strip():
        return False
    try:
        # 先尝试 HEAD（部分站点不支持）
        try:
            head = requests.head(url, allow_redirects=True, timeout=8)
            ctype = head.headers.get("Content-Type", "").lower()
        except Exception:
            ctype = ""

        # 如果 URL 以 .pdf 结尾或 HEAD 表示 PDF，直接 GET 保存
        if url.lower().endswith(".pdf") or "application/pdf" in ctype:
            r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and ("application/pdf" in r.headers.get("Content-Type", "").lower() or url.lower().endswith(".pdf")):
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                return False

        # 否则做 GET 再判断 content-type（有时候 GET 会返回 PDF）
        r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", "").lower():
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return True

        # 不是 PDF（按你的要求不解析 HTML 获取 PDF 链接）
        return False

    except Exception:
        # 任意异常视为失败
        return False

# --------- Public API ----------
def save_pdfs_from_url_list(urls: Iterable[Optional[str]],
                            outdir: str = "downloaded_pdfs",
                            overwrite: bool = False,
                            timeout: int = 20) -> List[Dict[str, Any]]:

    os.makedirs(outdir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    session = requests.Session()

    for url in urls:
        res = {"name": None, "url": url, "status": None, "path_or_msg": None}
        if not url:
            res.update({"status": "SKIP", "path_or_msg": "empty URL or None"})
            results.append(res)
            continue

        safe_name = make_safe_filename_from_url(url)
        if not safe_name:
            # fallback name
            h = hashlib.sha1(str(url).encode("utf-8")).hexdigest()[:12]
            safe_name = f"file_{h}.pdf"

        res["name"] = safe_name
        save_path = os.path.join(outdir, safe_name)

        if os.path.exists(save_path) and not overwrite:
            res.update({"status": "EXISTS", "path_or_msg": save_path})
            results.append(res)
            continue

        # 尝试下载（直接使用 requests）
        ok = download_pdf(url, save_path, timeout=timeout)
        if ok:
            res.update({"status": "OK", "path_or_msg": save_path})
        else:
            # 清理可能存在的残缺文件
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception:
                    pass
            res.update({"status": "FAIL", "path_or_msg": "could not download PDF or not a PDF"})
        results.append(res)

    return results

