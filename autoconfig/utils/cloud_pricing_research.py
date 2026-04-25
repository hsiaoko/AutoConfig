"""
Look up public cloud list prices for a resource spec using a search-augmented LLM.

Default backend: **Perplexity** ``/v1/sonar`` (web search + answer). Set
``PERPLEXITY_API_KEY`` (or pass ``api_key=``) from https://www.perplexity.ai/settings/api

This module uses only the standard library (``urllib``) for HTTPS.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

PERPLEXITY_URL = "https://api.perplexity.ai/v1/sonar"
DEFAULT_MODEL = "sonar"


def _num_or_zero(x: Any) -> float:
    if x is None:
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def attach_list_prices(cloud_pricing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add ``list_price_usd_per_hour`` and ``estimated_monthly_usd_730h`` for YAML.
    If no web/API result or no parsed JSON, both are 0.0.
    """
    out = dict(cloud_pricing)
    st = out.get("status")
    aj = out.get("answer_json")
    if st in ("skipped", "error"):
        h, m = 0.0, 0.0
    elif isinstance(aj, dict):
        h = _num_or_zero(aj.get("estimated_compute_usd_per_hour"))
        m = _num_or_zero(aj.get("estimated_monthly_usd_730h"))
    else:
        h, m = 0.0, 0.0
    out["list_price_usd_per_hour"] = h
    out["estimated_monthly_usd_730h"] = m
    if "currency" not in out:
        cur = (aj or {}).get("currency") if isinstance(aj, dict) else None
        out["currency"] = (cur or "USD") if isinstance(cur, str) else "USD"
    return out


def zero_cloud_pricing(reason: str) -> Dict[str, Any]:
    """No internet / no API: explicit zero prices (美元标价占位)."""
    return attach_list_prices(
        {
            "status": "unavailable",
            "engine": "none",
            "queried_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
    )


def _json_prompt(resource: Dict[str, Any]) -> str:
    spec = json.dumps(resource, ensure_ascii=False, indent=2)
    return f"""你是一名云厂商定价研究员。请使用联网检索，在 AWS、GCP、Azure 或阿里云/腾讯云 等**大型公有云**中，
找出**最接近**下列计算/存储资源规格的**当前公开价目表**（以按需 on-demand 或官方计算器标价为主，如有折扣请注明）。

如果找不到完全匹配，请给出**最接近的一档**实例/SKU 并说明差异。

**必须**只输出**一段合法 JSON 对象**（不要 Markdown，不要代码围栏），使用下列键（缺失用 null，数字用十进制）：
{{
  "suggested_provider": "AWS|GCP|Azure|Aliyun|Tencent|other|null",
  "suggested_instance_or_sku": "string 或 null",
  "list_price_model": "on_demand|spot|savings_plans|official_calculator|unknown",
  "estimated_compute_usd_per_hour": number 或 null,
  "estimated_monthly_usd_730h": number 或 null,
  "currency": "USD",
  "region_or_zone_example": "string 或 null",
  "storage_note": "与 storage_gb 相关的块存储/对象存储价格一句说明 或 null",
  "summary_zh": "2-4 句中文，说明如何对照规格与价格、主要不确定性",
  "caveats": "保留价/地区差/需登录等 或 null"
}}

资源配置如下：
{spec}
"""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}\s*$", text)
    if not m:
        m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    chunk = m.group(0)
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None


def _perplexity_request(
    *,
    api_key: str,
    model: str,
    user_text: str,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": user_text,
            }
        ],
        "temperature": 0.1,
        "search_mode": "web",
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        PERPLEXITY_URL,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _normalize_search_results(data: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in data.get("search_results") or []:
        if isinstance(item, dict) and item.get("url"):
            out.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                }
            )
    for u in data.get("citations") or []:
        if isinstance(u, str) and u:
            out.append({"title": "", "url": u})
    return out


def _message_text(data: Dict[str, Any]) -> str:
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: List[str] = []
        for p in c:
            if isinstance(p, dict) and p.get("text"):
                parts.append(str(p["text"]))
        return "\n".join(parts)
    return ""


def fetch_cloud_pricing_perplexity(
    resource: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Call Perplexity Sonar with web search; return a dict suitable for ``metadata.cloud_pricing``.
    """
    key = api_key or os.environ.get("PERPLEXITY_API_KEY") or os.environ.get(
        "CLOUD_PRICING_API_KEY"
    )
    mdl = model or os.environ.get("PERPLEXITY_MODEL") or DEFAULT_MODEL
    t0 = time.perf_counter()
    if not key:
        return attach_list_prices(
            {
                "status": "skipped",
                "engine": "perplexity",
                "reason": "未设置环境变量 PERPLEXITY_API_KEY（或传入 api_key）",
                "queried_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    try:
        raw_api = _perplexity_request(
            api_key=key, model=mdl, user_text=_json_prompt(resource), timeout=timeout
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
        return attach_list_prices(
            {
                "status": "error",
                "engine": "perplexity",
                "model": mdl,
                "error": str(e),
                "queried_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except json.JSONDecodeError as e:
        return attach_list_prices(
            {
                "status": "error",
                "engine": "perplexity",
                "model": mdl,
                "error": f"Invalid JSON from API: {e}",
                "queried_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    text = _message_text(raw_api)
    parsed = _extract_json_object(text) if text else None
    usage = raw_api.get("usage") or {}
    cost_info = usage.get("cost") if isinstance(usage, dict) else None
    api_cost: Optional[float] = None
    if isinstance(cost_info, dict) and "total_cost" in cost_info:
        try:
            api_cost = float(cost_info["total_cost"])
        except (TypeError, ValueError):
            api_cost = None

    elapsed_s = time.perf_counter() - t0
    out: Dict[str, Any] = {
        "status": "ok",
        "engine": "perplexity",
        "model": raw_api.get("model", mdl),
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(elapsed_s, 3),
        "answer_markdown": text,
        "answer_json": parsed,
        "search_sources": _normalize_search_results(raw_api),
        "api_usage": usage if usage else None,
        "api_request_cost_usd": api_cost,
    }
    if not parsed and text:
        out["status"] = "partial"
        out["parse_note"] = "模型返回无法解析为 JSON，已保留原文于 answer_markdown"
    return attach_list_prices(out)
