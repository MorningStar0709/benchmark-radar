from __future__ import annotations

import json
import re
import threading
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_query_surfaces import _catalog

from benchmark_radar.citation import BIBTEX_KEY
from benchmark_radar.models import RadarItem, RadarRun, SourceHealth
from benchmark_radar.query import QueryError, QueryPaths, QueryService
from benchmark_radar.query_cli import run_query_cli
from benchmark_radar.query_http import create_query_server
from benchmark_radar.related_work_render import latex_escape
from benchmark_radar.snapshots import write_snapshot


def _paths(tmp_path: Path) -> QueryPaths:
    """The shared query fixture plus a later day holding scholarly Radar leads."""
    paths = _catalog(tmp_path)
    generated_at = datetime(2026, 8, 30, 8, 0, tzinfo=UTC)
    write_snapshot(
        RadarRun(
            generated_at=generated_at,
            since=generated_at - timedelta(hours=48),
            items=[
                RadarItem(
                    source="arXiv",
                    source_id="2608.01234",
                    title="Agent Workbench Pro: Long-Horizon Agent Evaluation",
                    url="https://arxiv.org/abs/2608.01234",
                    published_at=generated_at - timedelta(hours=2),
                    summary=(
                        "Agents are everywhere. To address this gap, we introduce Agent "
                        "Workbench Pro, a benchmark of 50% harder tasks."
                    ),
                    authors=["Ada Lovelace", "Alan Turing"],
                    categories=["benchmark", "agentic"],
                ),
                RadarItem(
                    source="GitHub",
                    source_id="example/agent-workbench-fork",
                    title="Agent Workbench Fork",
                    url="https://github.com/example/agent-workbench-fork",
                    published_at=generated_at - timedelta(hours=3),
                    summary="A repository fork of the agent workbench harness.",
                    categories=["benchmark", "agentic"],
                ),
            ],
            health=[
                SourceHealth(source=source, ok=True, item_count=1, method="API")
                for source in ("arxiv", "github", "huggingface")
            ],
        ),
        paths.snapshots,
    )
    return paths


def _bib_keys(bibtex: str) -> list[str]:
    return re.findall(r"@misc\{([^,]+),", bibtex)


def _cited_keys(latex: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", latex):
        keys.update(key.strip() for key in group.split(","))
    return keys


def test_draft_cites_every_entry_and_the_radar_paper_once(tmp_path: Path) -> None:
    payload = QueryService(_paths(tmp_path)).related_work(["Agent benchmarks=agent workbench"])

    entry_keys = {entry["cite_key"] for entry in payload["entries"]}
    assert entry_keys, "fixture must retain at least one work"
    assert set(_bib_keys(payload["bibtex"])) == entry_keys | {BIBTEX_KEY}
    assert _cited_keys(payload["latex"]) == entry_keys | {BIBTEX_KEY}
    # The self-citation is quiet: one clause closing the last paragraph, after
    # every related work, never the opening line or a paragraph of its own.
    latex = payload["latex"]
    body = latex.split("\\section", 1)[1]
    assert latex.count(BIBTEX_KEY) == 1
    assert "Benchmark Radar" not in body
    assert all(latex.index(key) < latex.index(BIBTEX_KEY) for key in entry_keys)
    closing_paragraph = body.rsplit("\n\n", 1)[-1]
    assert closing_paragraph.startswith("\\paragraph{")
    assert BIBTEX_KEY in closing_paragraph.splitlines()[-1]
    assert _bib_keys(payload["bibtex"])[-1] == BIBTEX_KEY


def test_topics_keep_full_matches_unless_partial_is_requested(tmp_path: Path) -> None:
    service = QueryService(_paths(tmp_path))
    strict = service.related_work(["agent workbench"])
    loose = service.related_work(["agent workbench"], include_partial=True)

    strict_names = {entry["name"] for entry in strict["entries"]}
    assert "Science Discovery Suite" not in strict_names
    assert "New Agent Memory Benchmark" not in strict_names
    assert len(loose["entries"]) >= len(strict["entries"])


def test_radar_leads_are_scholarly_records_with_recorded_authors(tmp_path: Path) -> None:
    payload = QueryService(_paths(tmp_path)).related_work(["agent workbench"])
    by_name = {entry["name"]: entry for entry in payload["entries"]}

    assert "Agent Workbench Fork" not in by_name, "repository leads are not paper citations"
    lead = by_name["Agent Workbench Pro: Long-Horizon Agent Evaluation"]
    assert lead["cite_key"] == "lovelace2026agent"
    assert lead["arxiv_id"] == "2608.01234"
    assert "author       = {Ada Lovelace and Alan Turing}" in lead["bibtex"]
    assert "radar_lead_unverified" in lead["verification"]
    # The lead-in clause is dropped and the "we" sentence becomes an author citation.
    assert "\\citet{lovelace2026agent} introduce Agent Workbench Pro" in payload["latex"]
    assert "50\\% harder" in payload["latex"]


def test_catalog_records_without_authors_are_flagged_not_invented(tmp_path: Path) -> None:
    payload = QueryService(_paths(tmp_path)).related_work(["agent workbench"], include_radar=False)
    entry = next(item for item in payload["entries"] if item["name"] == "Agent Workbench")

    assert entry["authors"] == []
    assert "authors_missing" in entry["verification"]
    assert "  author " not in entry["bibtex"]
    assert "  key          = {Agent Workbench}," in entry["bibtex"]
    assert all(entry["kind"] == "catalog" for entry in payload["entries"])
    assert "radar_first_date" not in payload["coverage"]


def test_coverage_statement_names_the_corpus_window(tmp_path: Path) -> None:
    payload = QueryService(_paths(tmp_path)).related_work(["agent workbench"])

    coverage = payload["coverage"]
    assert coverage["radar_first_date"] == "2026-08-29"
    assert coverage["radar_latest_date"] == "2026-08-30"
    assert "not about the literature" in coverage["statement"]
    assert coverage["statement"] in payload["latex"]
    assert "| Work | Cite key |" in payload["markdown"]


def test_invalid_related_work_requests_are_machine_readable(tmp_path: Path) -> None:
    service = QueryService(_paths(tmp_path))
    with pytest.raises(QueryError) as empty:
        service.related_work(["Label="])
    assert empty.value.code == "invalid_query"
    with pytest.raises(QueryError) as limit:
        service.related_work(["agent"], per_topic=0)
    assert limit.value.code == "invalid_limit"


def test_latex_escape_handles_specials_and_greek() -> None:
    assert latex_escape("τ-bench 50% & $5") == "\\ensuremath{\\tau}-bench 50\\% \\& \\$5"
    assert latex_escape("ΔΑ") == "\\ensuremath{\\Delta}A"
    assert latex_escape("评测 bench ✅，ok") == "bench ,ok"


def test_cli_and_http_return_the_same_related_work_contract(tmp_path: Path, capsys) -> None:
    paths = _paths(tmp_path)
    tex_path, bib_path = tmp_path / "out" / "related.tex", tmp_path / "out" / "related.bib"
    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
            "science discovery",
            "--json",
            "--tex",
            str(tex_path),
            "--bib",
            str(bib_path),
            "--index",
            str(paths.index),
            "--shards",
            str(paths.shards),
            "--snapshots",
            str(paths.snapshots),
        ]
    )
    cli_payload = json.loads(capsys.readouterr().out)

    server = create_query_server(QueryService(paths), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        query = urllib.parse.urlencode(
            [("q", "Agent benchmarks=agent workbench"), ("q", "science discovery")]
        )
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/api/v1/related-work?{query}", timeout=5
        ) as response:
            http_payload = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert exit_code == 0
    assert cli_payload == http_payload
    assert [topic["label"] for topic in cli_payload["topics"]] == [
        "Agent benchmarks",
        "science discovery",
    ]
    assert tex_path.read_text(encoding="utf-8") == cli_payload["latex"]
    assert bib_path.read_text(encoding="utf-8") == cli_payload["bibtex"]
