"""Unit tests for the Task Manager CLI."""

import json
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

from app import (
    add,
    cli,
    complete,
    delete,
    edit,
    highlight_match,
    is_overdue,
    list_tasks,
    load_tasks,
    save_tasks,
    search_tasks,
    stats,
)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


class TestLoadTasks:
    def test_returns_empty_list_when_file_missing(self, isolated_tasks_file: Path) -> None:
        assert load_tasks() == []

    def test_loads_existing_tasks(self, isolated_tasks_file: Path) -> None:
        tasks = [{"id": 1, "name": "Test", "done": False}]
        isolated_tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
        assert load_tasks() == tasks

    def test_returns_empty_list_on_corrupt_file(self, isolated_tasks_file: Path) -> None:
        isolated_tasks_file.write_text("not-json", encoding="utf-8")
        assert load_tasks() == []

    def test_returns_empty_list_when_file_contains_object(self, isolated_tasks_file: Path) -> None:
        isolated_tasks_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        assert load_tasks() == []


class TestSaveTasks:
    def test_saves_and_reloads(self, isolated_tasks_file: Path) -> None:
        tasks = [{"id": 1, "name": "Test", "done": False}]
        save_tasks(tasks)
        assert load_tasks() == tasks

    def test_saves_empty_list(self, isolated_tasks_file: Path) -> None:
        save_tasks([])
        assert load_tasks() == []


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------


class TestIsOverdue:
    def test_past_due_date_is_overdue(self) -> None:
        task = {"done": False, "due_date": "2000-01-01"}
        assert is_overdue(task) is True

    def test_future_due_date_is_not_overdue(self) -> None:
        task = {"done": False, "due_date": "2099-12-31"}
        assert is_overdue(task) is False

    def test_done_task_is_never_overdue(self) -> None:
        task = {"done": True, "due_date": "2000-01-01"}
        assert is_overdue(task) is False

    def test_no_due_date_is_not_overdue(self) -> None:
        task = {"done": False, "due_date": None}
        assert is_overdue(task) is False

    def test_invalid_due_date_is_not_overdue(self) -> None:
        task = {"done": False, "due_date": "not-a-date"}
        assert is_overdue(task) is False


class TestSearchHelpers:
    def test_search_tasks_matches_name_case_insensitively(self, sample_tasks: list[dict]) -> None:
        matches = search_tasks(sample_tasks, "deploy")
        assert [task["id"] for task in matches] == [2]

    def test_search_tasks_matches_description_only(self, sample_tasks: list[dict]) -> None:
        matches = search_tasks(sample_tasks, "PIPELINE")
        assert [task["id"] for task in matches] == [2]

    def test_search_tasks_sorts_by_priority_then_id(self) -> None:
        tasks = [
            {"id": 4, "name": "Deploy low", "priority": "low", "done": False},
            {"id": 3, "name": "Deploy high later", "priority": "high", "done": False},
            {"id": 1, "name": "Deploy medium", "priority": "medium", "done": False},
            {"id": 2, "name": "Deploy high earlier", "priority": "high", "done": False},
        ]

        matches = search_tasks(tasks, "deploy")

        assert [task["id"] for task in matches] == [2, 3, 1, 4]

    def test_highlight_match_preserves_text_and_styles_matches(self) -> None:
        text = highlight_match("Deploy then deploy", "deploy")

        assert text.plain == "Deploy then deploy"
        assert [(span.start, span.end, str(span.style)) for span in text.spans] == [
            (0, 6, "bold yellow"),
            (12, 18, "bold yellow"),
        ]

    def test_highlight_match_treats_keyword_as_literal_text(self) -> None:
        text = highlight_match("Fix a.b task", "a.b")

        assert text.plain == "Fix a.b task"
        assert [(span.start, span.end, str(span.style)) for span in text.spans] == [
            (4, 7, "bold yellow"),
        ]


# ---------------------------------------------------------------------------
# CLI commands via CliRunner
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestAddCommand:
    def test_add_creates_task(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["add", "Buy milk"])
        assert result.exit_code == 0
        tasks = load_tasks()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "Buy milk"
        assert tasks[0]["done"] is False

    def test_add_sets_defaults(self, runner: CliRunner) -> None:
        runner.invoke(cli, ["add", "Task"])
        task = load_tasks()[0]
        assert task["priority"] == "medium"
        assert task["tags"] == []
        assert task["due_date"] is None
        assert task["description"] == ""
        assert "created_at" in task

    def test_add_with_options(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            ["add", "Deploy", "--priority", "high", "--due", "2099-12-31", "--tag", "work", "--tag", "devops"],
        )
        assert result.exit_code == 0
        task = load_tasks()[0]
        assert task["priority"] == "high"
        assert task["due_date"] == "2099-12-31"
        assert task["tags"] == ["work", "devops"]

    def test_add_empty_name_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["add", "   "])
        assert result.exit_code != 0
        assert load_tasks() == []

    def test_add_name_too_long_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["add", "x" * 201])
        assert result.exit_code != 0

    def test_add_invalid_due_date_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["add", "Task", "--due", "31-12-2099"])
        assert result.exit_code != 0
        assert load_tasks() == []

    def test_add_assigns_sequential_ids(self, runner: CliRunner) -> None:
        runner.invoke(cli, ["add", "First"])
        runner.invoke(cli, ["add", "Second"])
        ids = [t["id"] for t in load_tasks()]
        assert ids == [1, 2]


class TestListCommand:
    def test_list_empty(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_list_shows_tasks(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "Buy groceries" in result.output
        assert "Deploy to production" in result.output

    def test_list_filter_pending(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--status", "pending"])
        assert result.exit_code == 0
        assert "Write unit tests" not in result.output  # it's done

    def test_list_filter_done(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--status", "done"])
        assert result.exit_code == 0
        assert "Write unit tests" in result.output
        assert "Buy groceries" not in result.output

    def test_list_filter_priority(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--priority", "high"])
        assert result.exit_code == 0
        assert "Deploy to production" in result.output
        assert "Buy groceries" not in result.output

    def test_list_filter_tag(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--tag", "personal"])
        assert result.exit_code == 0
        assert "Buy groceries" in result.output
        assert "Deploy to production" not in result.output

    def test_list_filter_overdue(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--overdue"])
        assert result.exit_code == 0
        assert "Deploy to production" in result.output  # due 2020-01-01
        assert "Buy groceries" not in result.output

    def test_list_no_match_message(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["list", "--tag", "nonexistent"])
        assert result.exit_code == 0
        assert "No tasks match" in result.output


class TestSearchCommand:
    def test_search_help_describes_keyword_and_example(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["search", "--help"])

        assert result.exit_code == 0
        assert "KEYWORD" in result.output
        assert 'python app.py search "deploy"' in result.output

    def test_search_shows_pending_and_done_statuses(
        self, runner: CliRunner, isolated_tasks_file: Path
    ) -> None:
        tasks = [
            {
                "id": 1,
                "name": "Deploy API",
                "description": "",
                "priority": "high",
                "done": False,
            },
            {
                "id": 2,
                "name": "Document deploy",
                "description": "",
                "priority": "medium",
                "done": True,
            },
        ]
        isolated_tasks_file.write_text(json.dumps(tasks), encoding="utf-8")

        result = runner.invoke(cli, ["search", "deploy"])

        assert result.exit_code == 0
        assert "Deploy API" in result.output
        assert "Document deploy" in result.output
        assert "Pending" in result.output
        assert "Done" in result.output

    def test_search_displays_results_sorted_by_priority_then_id(
        self, runner: CliRunner, isolated_tasks_file: Path
    ) -> None:
        tasks = [
            {"id": 4, "name": "Deploy low", "priority": "low", "done": False},
            {"id": 3, "name": "Deploy high later", "priority": "high", "done": False},
            {"id": 1, "name": "Deploy medium", "priority": "medium", "done": False},
            {"id": 2, "name": "Deploy high earlier", "priority": "high", "done": False},
        ]
        isolated_tasks_file.write_text(json.dumps(tasks), encoding="utf-8")

        result = runner.invoke(cli, ["search", "deploy"])

        assert result.exit_code == 0
        assert result.output.index("Deploy high earlier") < result.output.index("Deploy high later")
        assert result.output.index("Deploy high later") < result.output.index("Deploy medium")
        assert result.output.index("Deploy medium") < result.output.index("Deploy low")

    def test_search_no_match_message_exits_successfully(
        self, runner: CliRunner, sample_tasks: list[dict]
    ) -> None:
        result = runner.invoke(cli, ["search", "missing"])

        assert result.exit_code == 0
        assert "No tasks match your search." in result.output

    def test_search_blank_keyword_fails_without_listing_tasks(
        self, runner: CliRunner, sample_tasks: list[dict]
    ) -> None:
        result = runner.invoke(cli, ["search", "   "])

        assert result.exit_code != 0
        assert "Search keyword cannot be empty" in result.output
        assert "Buy groceries" not in result.output

    def test_search_handles_missing_description(self, runner: CliRunner, isolated_tasks_file: Path) -> None:
        tasks = [{"id": 1, "name": "Deploy app", "priority": "low", "done": False}]
        isolated_tasks_file.write_text(json.dumps(tasks), encoding="utf-8")

        result = runner.invoke(cli, ["search", "deploy"])

        assert result.exit_code == 0
        assert "Deploy app" in result.output
        assert "—" in result.output


class TestCompleteCommand:
    def test_complete_marks_done(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["complete", "1"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["done"] is True

    def test_complete_already_done(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["complete", "3"])  # task 3 is already done
        assert result.exit_code == 0
        assert "already" in result.output

    def test_complete_nonexistent_id_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["complete", "999"])
        assert result.exit_code != 0


class TestEditCommand:
    def test_edit_name(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--name", "New name"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["name"] == "New name"

    def test_edit_priority(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--priority", "high"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["priority"] == "high"

    def test_edit_due_date(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--due", "2099-06-30"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["due_date"] == "2099-06-30"

    def test_edit_clear_due_date(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "2", "--due", ""])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 2)
        assert task["due_date"] is None

    def test_edit_tags(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--tag", "home", "--tag", "urgent"])
        assert result.exit_code == 0
        task = next(t for t in load_tasks() if t["id"] == 1)
        assert task["tags"] == ["home", "urgent"]

    def test_edit_no_changes_warns(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1"])
        assert result.exit_code == 0
        assert "No changes" in result.output

    def test_edit_nonexistent_id_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "999", "--name", "Nope"])
        assert result.exit_code != 0

    def test_edit_invalid_due_date_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["edit", "1", "--due", "bad-date"])
        assert result.exit_code != 0


class TestDeleteCommand:
    def test_delete_removes_task(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["delete", "1"])
        assert result.exit_code == 0
        ids = [t["id"] for t in load_tasks()]
        assert 1 not in ids

    def test_delete_nonexistent_id_fails(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["delete", "999"])
        assert result.exit_code != 0


class TestStatsCommand:
    def test_stats_empty(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "0" in result.output

    def test_stats_counts(self, runner: CliRunner, sample_tasks: list[dict]) -> None:
        result = runner.invoke(cli, ["stats"])
        assert result.exit_code == 0
        assert "3" in result.output  # total
