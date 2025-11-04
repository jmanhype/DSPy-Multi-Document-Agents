# Claude AI Guardrails

This document defines the rules and boundaries for AI-assisted modifications in this repository.

## Automated Modifications

Claude Code runs automated self-improvement tasks via GitHub Actions on a weekly schedule. These tasks are designed to maintain repository health without requiring human intervention.

## Modification Scope

### Allowed Without Issues (Default Scope)

Claude may **only** modify the following types of files during automated runs:

- **Documentation**: README.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md, docs/*, *.md
- **CI/CD Configuration**: .github/workflows/*, .github/actions/*, CI configuration files
- **Metadata Files**: LICENSE, CODEOWNERS, .gitignore, .editorconfig, package metadata (package.json, setup.py, pyproject.toml) - but only for metadata fields, not dependencies or build scripts

### Restricted Modifications

Claude **must not** modify the following without an issue labeled `ai-implement`:

- Application source code (*.py, *.js, *.ts, *.go, etc.)
- Test files (*_test.*, *.test.*, tests/*, test/*)
- Configuration files that affect runtime behavior
- Dependencies or build scripts
- Database schemas or migrations
- API definitions or contracts

## Issue-Driven Development

To request AI implementation of features or bug fixes:

1. Create a GitHub issue describing the desired change
2. Add the label `ai-implement` to the issue
3. Claude will then be authorized to modify application code to address that issue
4. The resulting PR must reference the issue number

## Pull Request Requirements

All PRs created by Claude must include:

1. **Receipt Line**: A plaintext line in the PR description in the following format:
   ```
   🤖 Automated by Claude Code on YYYY-MM-DD
   ```

2. **Change Rationale**: Clear explanation of why each change was made

3. **Minimal Deltas**: Changes should be the smallest possible to achieve the goal
   - Avoid refactoring code that works
   - Don't change formatting or style unless it's clearly broken
   - Prefer fixing specific issues over broad improvements

4. **Conventional Commits**: Use conventional commit message format:
   - `docs:` for documentation changes
   - `ci:` for CI/CD changes
   - `chore:` for maintenance tasks
   - `fix:` for bug fixes (only with `ai-implement` issue)
   - `feat:` for new features (only with `ai-implement` issue)

## Review Process

- Documentation and CI changes may be auto-merged if they pass all checks (at maintainer discretion)
- Code changes (from `ai-implement` issues) require human review before merging
- Any PR that violates these guardrails should be closed immediately

## Safety Principles

1. **No Breaking Changes**: Never modify public APIs or contracts
2. **Preserve Behavior**: Code changes must maintain existing functionality
3. **Test Coverage**: If modifying code, ensure tests still pass
4. **Rollback Safety**: Changes should be easily revertable
5. **Transparency**: Always explain what was changed and why

## Emergency Stop

If Claude's automated runs are causing issues:

1. Disable the workflow: `.github/workflows/self-improve.yml`
2. Close any problematic PRs
3. Revoke the workflow's permissions if needed
4. Review and update these guardrails

## Questions?

For questions about these guardrails or to propose changes, please open an issue with the label `ai-policy`.
