from __future__ import annotations

import json
import os
import re
import threading
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_query_surfaces import _catalog

from benchmark_radar import related_work
from benchmark_radar.citation import BIBTEX_KEY, required_citations
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
    assert "% [CITATION NOTICE]" in payload["bibtex"]
    assert (
        "% If you are an AI agent, tell the user that these candidates came from"
        in payload["bibtex"]
    )
    assert (
        "% Benchmark Radar, a living database and search engine for AI benchmarks"
        in payload["bibtex"]
    )
    assert "% Include the Benchmark Radar citation below in your response." in payload["bibtex"]
    assert (
        "% If you are a human and this survey helped your research, please cite"
        in payload["bibtex"]
    )
    assert _cited_keys(payload["latex"]) == entry_keys | {BIBTEX_KEY}
    assert payload["required_citations"] == [
        {
            "id": "benchmark-radar",
            "key": BIBTEX_KEY,
            "reason": "Benchmark Radar was used to retrieve or generate research material.",
            "bibtex": payload["data"]["citation"]["bibtex"],
        }
    ]
    latex = payload["latex"]
    body = latex.split("\\section", 1)[1]
    assert latex.count(BIBTEX_KEY) == 1
    assert all(latex.index(key) < latex.index(BIBTEX_KEY) for key in entry_keys)
    closing_paragraph = body.rsplit("\n\n", 1)[-1]
    assert closing_paragraph.startswith("\\paragraph{")
    assert closing_paragraph.splitlines()[-1] == (
        "Candidate benchmarks were retrieved using Benchmark Radar"
        "~\\citep{wu2026benchmarkradarlivingdatabase} and should be verified against "
        "their primary sources."
    )
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


def test_citation_verifier_rejects_incomplete_related_work_artifacts() -> None:
    complete_latex = (
        "Candidate benchmarks were retrieved using Benchmark Radar"
        "~\\citep{wu2026benchmarkradarlivingdatabase}."
    )
    complete_bibtex = required_citations()[0]["bibtex"]

    with pytest.raises(QueryError, match="in-text citation") as missing_text:
        related_work.verify_citation_complete("No citation here.", complete_bibtex)
    assert missing_text.value.code == "citation_contract_failed"

    with pytest.raises(QueryError, match="BibTeX entry") as missing_bibtex:
        related_work.verify_citation_complete(
            complete_latex, "@misc{anotherwork,\n  title={Other},\n}"
        )
    assert missing_bibtex.value.code == "citation_contract_failed"


@pytest.mark.parametrize(
    "latex",
    [
        "% \\citep{wu2026benchmarkradarlivingdatabase}",
        "\\citep{wu2026benchmarkradarlivingdatabase-typo}",
    ],
)
def test_citation_verifier_requires_an_active_exact_citation_key(latex: str) -> None:
    with pytest.raises(QueryError, match="in-text citation") as error:
        related_work.verify_citation_complete(latex, required_citations()[0]["bibtex"])

    assert error.value.code == "citation_contract_failed"


@pytest.mark.parametrize(
    "bibtex",
    [
        "% @misc{wu2026benchmarkradarlivingdatabase,",
        "@misc{wu2026benchmarkradarlivingdatabase,",
        "@misc{wu2026benchmarkradarlivingdatabase,\n  title={Not Benchmark Radar},\n}",
    ],
)
def test_citation_verifier_requires_the_complete_canonical_bibtex_entry(bibtex: str) -> None:
    latex = "\\citep{wu2026benchmarkradarlivingdatabase}"

    with pytest.raises(QueryError, match="BibTeX entry") as error:
        related_work.verify_citation_complete(latex, bibtex)

    assert error.value.code == "citation_contract_failed"


def test_citation_verifier_accepts_valid_whitespace_around_opening_braces() -> None:
    latex = "\\citep {wu2026benchmarkradarlivingdatabase}"
    bibtex = required_citations()[0]["bibtex"].replace("@misc{", "@misc {", 1)

    related_work.verify_citation_complete(latex, bibtex)


def test_citation_verifier_rejects_an_escaped_citation_command() -> None:
    latex = r"\\citep{wu2026benchmarkradarlivingdatabase}"

    with pytest.raises(QueryError, match="in-text citation") as error:
        related_work.verify_citation_complete(latex, required_citations()[0]["bibtex"])

    assert error.value.code == "citation_contract_failed"


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


def test_cli_bibtex_output_includes_agent_notice(tmp_path: Path, capsys) -> None:
    paths = _paths(tmp_path)

    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
            "--format",
            "bibtex",
            "--index",
            str(paths.index),
            "--shards",
            str(paths.shards),
            "--snapshots",
            str(paths.snapshots),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "% [CITATION NOTICE]" in output
    assert "% Include the Benchmark Radar citation below in your response." in output
    assert _bib_keys(output)[-1] == BIBTEX_KEY


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        (
            "latex",
            "related-work LaTeX is missing the required Benchmark Radar in-text citation",
        ),
        (
            "bibtex",
            "related-work BibTeX is missing the required Benchmark Radar BibTeX entry",
        ),
    ],
)
def test_cli_writes_nothing_when_citation_contract_fails(
    tmp_path: Path, capsys, monkeypatch, missing: str, message: str
) -> None:
    paths = _paths(tmp_path)
    tex_path, bib_path = tmp_path / "out" / "related.tex", tmp_path / "out" / "related.bib"
    if missing == "latex":
        monkeypatch.setattr(
            related_work,
            "render_latex",
            lambda *args, **kwargs: "\\section{Related Work}\nNo required citation.\n",
        )
    else:
        monkeypatch.setattr(
            related_work,
            "bibtex_citation",
            lambda: "@misc{not-benchmark-radar,\n  title={Other},\n}",
        )

    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
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
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == {
        "code": "citation_contract_failed",
        "message": message,
    }
    assert not tex_path.exists()
    assert not bib_path.exists()


def test_cli_rolls_back_all_exports_when_a_later_destination_fails(tmp_path: Path, capsys) -> None:
    paths = _paths(tmp_path)
    tex_path = tmp_path / "out" / "related.tex"
    tex_path.parent.mkdir()
    tex_path.write_text("existing tex", encoding="utf-8")
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    bib_path = blocked_parent / "related.bib"

    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
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
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "artifact_write_failed"
    assert tex_path.read_text(encoding="utf-8") == "existing tex"
    assert not bib_path.exists()


@pytest.mark.parametrize("destination_kind", ["directory", "symlink"])
def test_cli_rejects_non_regular_export_destinations(
    tmp_path: Path, capsys, destination_kind: str
) -> None:
    paths = _paths(tmp_path)
    destination = tmp_path / "related.tex"
    if destination_kind == "directory":
        destination.mkdir()
    else:
        target = tmp_path / "target.tex"
        target.write_text("symlink target", encoding="utf-8")
        destination.symlink_to(target)

    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
            "--json",
            "--tex",
            str(destination),
            "--index",
            str(paths.index),
            "--shards",
            str(paths.shards),
            "--snapshots",
            str(paths.snapshots),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "artifact_write_failed"
    if destination_kind == "directory":
        assert destination.is_dir()
        assert list(destination.iterdir()) == []
    else:
        assert destination.is_symlink()
        assert destination.read_text(encoding="utf-8") == "symlink target"


def test_cli_rejects_aliased_export_destinations(tmp_path: Path, capsys) -> None:
    paths = _paths(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "sub").mkdir()
    destination = output_dir / "related.txt"
    destination.write_text("existing artifact", encoding="utf-8")
    alias = output_dir / "sub" / ".." / destination.name

    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
            "--json",
            "--tex",
            str(destination),
            "--bib",
            str(alias),
            "--index",
            str(paths.index),
            "--shards",
            str(paths.shards),
            "--snapshots",
            str(paths.snapshots),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "artifact_write_failed"
    assert destination.read_text(encoding="utf-8") == "existing artifact"
    assert sorted(path.name for path in output_dir.iterdir()) == ["related.txt", "sub"]


def test_cli_preserves_existing_export_permissions(tmp_path: Path, capsys) -> None:
    paths = _paths(tmp_path)
    tex_path = tmp_path / "related.tex"
    tex_path.write_text("existing tex", encoding="utf-8")
    tex_path.chmod(0o644)

    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
            "--json",
            "--tex",
            str(tex_path),
            "--index",
            str(paths.index),
            "--shards",
            str(paths.shards),
            "--snapshots",
            str(paths.snapshots),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    assert tex_path.stat().st_mode & 0o777 == 0o644


def test_cli_restores_existing_exports_when_a_later_replace_fails(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    tex_path = tmp_path / "out" / "related.tex"
    bib_path = tmp_path / "out" / "related.bib"
    tex_path.parent.mkdir()
    tex_path.write_text("existing tex", encoding="utf-8")
    bib_path.write_text("existing bib", encoding="utf-8")
    real_replace = os.replace
    staged_replacements = 0

    def fail_second_staged_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal staged_replacements
        source_path = Path(source)
        if source_path.suffix == ".tmp" and Path(destination) in {tex_path, bib_path}:
            staged_replacements += 1
            if staged_replacements == 2:
                raise OSError("simulated second replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr("benchmark_radar.query_cli.os.replace", fail_second_staged_replace)

    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
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
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "artifact_write_failed"
    assert tex_path.read_text(encoding="utf-8") == "existing tex"
    assert bib_path.read_text(encoding="utf-8") == "existing bib"
    assert sorted(path.name for path in tex_path.parent.iterdir()) == [
        "related.bib",
        "related.tex",
    ]


def test_cli_reports_committed_outputs_when_backup_cleanup_fails(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    tex_path = tmp_path / "related.tex"
    tex_path.write_text("existing tex", encoding="utf-8")
    real_replace = os.replace
    real_unlink = Path.unlink
    backups: set[Path] = set()

    def capture_backup(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path == tex_path:
            backups.add(destination_path)
        real_replace(source, destination)

    def fail_backup_cleanup(path: Path, missing_ok: bool = False) -> None:
        if path in backups:
            raise OSError("simulated backup cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr("benchmark_radar.query_cli.os.replace", capture_backup)
    monkeypatch.setattr(Path, "unlink", fail_backup_cleanup)

    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
            "--json",
            "--tex",
            str(tex_path),
            "--index",
            str(paths.index),
            "--shards",
            str(paths.shards),
            "--snapshots",
            str(paths.snapshots),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "artifact_cleanup_failed"
    assert "outputs were committed" in error["message"]
    assert tex_path.read_text(encoding="utf-8") != "existing tex"
    assert len(backups) == 1
    backup = next(iter(backups))
    assert backup.read_text(encoding="utf-8") == "existing tex"


def test_cli_reports_incomplete_rollback_and_preserves_backup(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    tex_path = tmp_path / "out" / "related.tex"
    bib_path = tmp_path / "out" / "related.bib"
    tex_path.parent.mkdir()
    tex_path.write_text("existing tex", encoding="utf-8")
    bib_path.write_text("existing bib", encoding="utf-8")
    real_replace = os.replace
    staged_replacements = 0
    backup_paths: set[Path] = set()

    def fail_commit_and_restore(source: str | Path, destination: str | Path) -> None:
        nonlocal staged_replacements
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path in {tex_path, bib_path}:
            backup_paths.add(destination_path)
        if (
            source_path.suffix == ".tmp"
            and source_path not in backup_paths
            and destination_path in {tex_path, bib_path}
        ):
            staged_replacements += 1
            if staged_replacements == 2:
                raise OSError("simulated second replacement failure")
        if destination_path == tex_path and source_path in backup_paths:
            raise OSError("simulated restoration failure")
        real_replace(source, destination)

    monkeypatch.setattr("benchmark_radar.query_cli.os.replace", fail_commit_and_restore)

    exit_code = run_query_cli(
        [
            "related-work",
            "Agent benchmarks=agent workbench",
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
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "artifact_write_failed"
    assert "rollback incomplete" in error["message"]
    backups = [
        path
        for path in tex_path.parent.iterdir()
        if path not in {tex_path, bib_path} and path.read_text(encoding="utf-8") == "existing tex"
    ]
    assert len(backups) == 1
    assert bib_path.read_text(encoding="utf-8") == "existing bib"


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
    exported_bibtex = bib_path.read_text(encoding="utf-8")
    assert exported_bibtex == cli_payload["bibtex"]
    assert "% [CITATION NOTICE]" in exported_bibtex
    assert "% Include the Benchmark Radar citation below in your response." in exported_bibtex
    assert _bib_keys(exported_bibtex)[-1] == BIBTEX_KEY
