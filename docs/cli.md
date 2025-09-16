# CLI

## Revoice UI

Запустить озвучку одной фразы:
```bash
uv run python -m ui.main --say "текст"
```

Параметры:
```text
usage: - [-h] [--say SAY]

Revoice UI

options:
  -h, --help  show this help message and exit
  --say SAY   Text to synthesize and exit
```

## revoice-check (QA)

Команда агрегирует установку dev-зависимостей и запуск `ruff`, `mypy`, `pytest`,
поддерживая вывод в stdout/JSON/Markdown и сохранение отчёта:

```bash
uv run revoice-check --format markdown --output reports/qa_report.md
```

Помощь по параметрам:

```text
usage: revoice-check [-h] [--format {stdout,json,markdown}] [--output OUTPUT]

Run QA checks with uv helpers

options:
  -h, --help            show this help message and exit
  --format {stdout,json,markdown}
                        Output format for the report
  --output OUTPUT       Optional file path to store the generated report
```
