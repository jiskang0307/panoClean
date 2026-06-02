"""
evaluate.py — 배치 처리 결과 통계 계산 및 HTML 리포트 생성.

사용:
    from evaluate import generate_batch_report, compute_batch_stats
    stats = compute_batch_stats(results)
    generate_batch_report(results, "output/report.html", stats)
"""

from __future__ import annotations

import html
from pathlib import Path


def compute_batch_stats(results: list[dict]) -> dict:
    """
    배치 전체 통계 계산.

    반환:
      {
        "total":               int,
        "coverage_avg":        float,   # down face 제외 평균
        "inpaint_ratio":       float,   # inpainting 사용 비율
        "no_photographer_ratio": float, # PHOTOGRAPHER 미검출 비율
        "elapsed_avg":         float,   # 평균 처리 시간 (초)
        "elapsed_total":       float,   # 누적 처리 시간 (초)
        "background_persons":  int,     # 모자이크 처리된 배경 인원 합계
      }
    """
    if not results:
        return {}

    n = len(results)
    coverages:    list[float] = []
    inpaints:     list[bool]  = []
    no_photo:     int         = 0
    elapsed_list: list[float] = []
    total_bg:     int         = 0

    for r in results:
        elapsed_list.append(r.get("elapsed_sec", 0.0))
        total_bg += r.get("background_persons", 0)

        face_stats: dict = r.get("faces", {})
        any_photographer = False
        for fname, fs in face_stats.items():
            if fname == "down":
                continue
            if fs.get("photographer"):
                any_photographer = True
                coverages.append(fs.get("coverage", 0.0))
                inpaints.append(bool(fs.get("did_inpaint", False)))

        if not any_photographer:
            no_photo += 1

    return {
        "total":                 n,
        "coverage_avg":          sum(coverages) / len(coverages) if coverages else 0.0,
        "inpaint_ratio":         sum(inpaints) / len(inpaints) if inpaints else 0.0,
        "no_photographer_ratio": no_photo / n,
        "elapsed_avg":           sum(elapsed_list) / n,
        "elapsed_total":         sum(elapsed_list),
        "background_persons":    total_bg,
    }


def generate_batch_report(
    results: list[dict],
    output_path: str,
    stats: dict | None = None,
) -> None:
    """
    HTML 리포트를 output_path에 저장.

    내용:
      - 전체 처리 통계 요약
      - 이미지별 상세 테이블
      - PHOTOGRAPHER 미검출 경고 목록
    """
    if stats is None:
        stats = compute_batch_stats(results)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    warnings: list[str] = []

    for r in results:
        fname       = html.escape(r.get("target", ""))
        elapsed     = f"{r.get('elapsed_sec', 0):.1f}s"
        cov_avg     = f"{r.get('coverage_avg', 0):.2f}"
        bg_persons  = str(r.get("background_persons", 0))
        face_stats  = r.get("faces", {})

        face_cells = []
        any_photo  = False
        for fn in ["front", "right", "back", "left", "up", "down"]:
            fs = face_stats.get(fn, {})
            if fn == "down":
                method = html.escape(fs.get("method", ""))
                has_p  = "✓" if fs.get("photographer") else "—"
                face_cells.append(f'<td title="down">{method} {has_p}</td>')
            else:
                if not fs.get("photographer"):
                    face_cells.append("<td>—</td>")
                else:
                    any_photo = True
                    cov   = fs.get("coverage", 0)
                    inpnt = "✓" if fs.get("did_inpaint") else "·"
                    cls   = "ok" if cov >= 0.7 else ("warn" if cov >= 0.4 else "bad")
                    face_cells.append(
                        f'<td class="{cls}">{cov:.2f} {inpnt}</td>'
                    )

        if not any_photo:
            warnings.append(fname)
            row_cls = ' class="no-photo"'
        else:
            row_cls = ""

        cells = "".join(face_cells)
        rows.append(
            f"<tr{row_cls}>"
            f"<td>{fname}</td>{cells}"
            f"<td>{cov_avg}</td>"
            f"<td>{bg_persons}</td>"
            f"<td>{elapsed}</td>"
            f"</tr>"
        )

    summary_html = ""
    if stats:
        summary_html = f"""
        <table class="summary">
          <tr><th>총 이미지</th><td>{stats.get('total', 0)}</td></tr>
          <tr><th>평균 coverage</th><td>{stats.get('coverage_avg', 0):.3f}</td></tr>
          <tr><th>inpainting 사용률</th><td>{stats.get('inpaint_ratio', 0):.1%}</td></tr>
          <tr><th>PHOTOGRAPHER 미검출률</th><td>{stats.get('no_photographer_ratio', 0):.1%}</td></tr>
          <tr><th>평균 처리 시간</th><td>{stats.get('elapsed_avg', 0):.1f}s</td></tr>
          <tr><th>총 처리 시간</th><td>{stats.get('elapsed_total', 0):.1f}s</td></tr>
          <tr><th>배경 인물 모자이크</th><td>{stats.get('background_persons', 0)}명</td></tr>
        </table>
        """

    warn_html = ""
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warn_html = f'<div class="warn-box"><b>⚠ PHOTOGRAPHER 미검출 ({len(warnings)}장)</b><ul>{items}</ul></div>'

    rows_html = "\n".join(rows)
    page = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>처리 리포트</title>
<style>
  body {{ font-family: sans-serif; font-size: 13px; margin: 20px; }}
  h1   {{ font-size: 18px; }}
  h2   {{ font-size: 14px; margin-top: 24px; }}
  table {{ border-collapse: collapse; margin-bottom: 16px; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: center; }}
  th {{ background: #f0f0f0; }}
  tr:hover {{ background: #f8f8ff; }}
  td.ok   {{ background: #d4edda; }}
  td.warn {{ background: #fff3cd; }}
  td.bad  {{ background: #f8d7da; }}
  tr.no-photo td {{ color: #888; }}
  .summary th {{ text-align: left; }}
  .warn-box {{ background: #fff3cd; border: 1px solid #ffc107;
               padding: 8px 12px; border-radius: 4px; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>360° 사람 제거 처리 리포트</h1>
{summary_html}
{warn_html}
<h2>이미지별 결과</h2>
<p>coverage 셀: 배경 교체 커버리지 (✓=inpainting 추가 적용, ·=스킵). 색상: 녹색≥0.7 / 주황≥0.4 / 빨강&lt;0.4</p>
<table>
  <tr>
    <th>파일명</th>
    <th>front</th><th>right</th><th>back</th><th>left</th><th>up</th><th>down</th>
    <th>cov_avg</th><th>배경인물</th><th>시간</th>
  </tr>
  {rows_html}
</table>
</body>
</html>
"""
    Path(output_path).write_text(page, encoding="utf-8")
