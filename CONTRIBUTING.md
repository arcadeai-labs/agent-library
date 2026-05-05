# Contributing to Agent Library

Thanks for your interest in contributing. Agent Library is a community project from Arcade.dev engineers — issues and pull requests are monitored on a best-effort basis. See the [README](README.md#support) for the full support story.

## Filing issues

Before opening an issue, please:

- Search existing issues to see if it's already been reported.
- Include the version (`libr --version` or the `agent-library` version from `pip show`), Python version, and OS.
- For bugs, include a minimal reproduction (commands run, files involved, observed vs. expected behavior).
- For feature requests, describe the use case before the proposed solution. The use case is usually the more useful half.

## Submitting pull requests

1. **Open an issue first** for non-trivial changes. A quick "I'd like to do X — does that fit?" comment saves both sides time. Small fixes (typos, obvious bugs, doc clarifications) can skip this.
2. **Fork and branch.** Branch off `main`. Keep PRs scoped — one logical change per PR. Mixed PRs (refactor + feature + drive-by cleanup) are hard to review and usually get split.
3. **Match the existing style.** Run `make check` (lint, format, typecheck) and `make test` before pushing. CI runs the same checks.
4. **Tests.** New behavior needs tests. Bug fixes should include a regression test that fails before the fix and passes after. Slow tests that load real embedding models should be marked `@pytest.mark.slow`.
5. **Commits.** Conventional commit prefixes (`feat:`, `fix:`, `chore:`, `docs:`) are appreciated but not required. Keep commit messages focused on the *why*.
6. **PR description.** Explain what changed and why. Link the issue if there is one.

## Development setup

```bash
git clone <your fork>
cd librarian
./setup.sh
make install
make test
```

See [CLAUDE.md](CLAUDE.md) for a fuller architecture and development guide — the same doc agents use to navigate the codebase.

## What we'll likely accept

- Bug fixes with regression tests.
- New parsers for additional file types.
- New embedding providers behind the existing `EmbeddingProvider` interface.
- Documentation improvements.
- Performance fixes with before/after numbers.

## What we'll likely push back on

- Large architectural rewrites without prior discussion.
- New top-level dependencies (we keep the dependency surface small on purpose — most things should be optional extras).
- Features that only work in a specific hosted environment.
- Removing the read/write tool split (this is intentional — see the launch blog post for the design rationale).

## Response times

This is a side project for the engineers who maintain it. Expect best-effort response times measured in days-to-weeks, not hours. If something is urgent and security-related, follow [SECURITY.md](SECURITY.md) instead — that path is monitored more closely.
