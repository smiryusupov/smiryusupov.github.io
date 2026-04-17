# smiryusupov.github.io

Minimal Quarto site for working notes and experiments. Rendered to `docs/` for GitHub Pages.

## Structure

```
_quarto.yml           site config
_nav.html             custom navigation (injected via include-before-body)
index.qmd             homepage
about.qmd             about page
styles.css            all styling
blog/
  index.qmd           writing index (chronological listing)
  posts/              post files (.qmd)
docs/                 rendered output (GitHub Pages)
```

## Local preview

```bash
quarto preview
```

## Render

```bash
quarto render
```

## Publish on GitHub Pages

1. Push to `main`
2. Settings → Pages → Deploy from branch → `main` / `/docs`

## Add a post

1. Create `blog/posts/your-post.qmd`
2. Add front matter — `title`, `date`, `description`, and one or more `categories`
3. Run `quarto render`

Example front matter:

```yaml
---
title: "Post title"
date: 2026-05-01
description: "One sentence."
categories: [notes]
---
```

## Topic taxonomy

Posts are tagged with one or more of these terms. Categories are currently hidden from the public listing — they exist in front matter only, as internal structure for future use.

| Term                  | Use for                                              |
|-----------------------|------------------------------------------------------|
| `notes`               | Short observations, references, working thoughts     |
| `experiments`         | Computational or analytical experiments with results |
| `statistics`          | Statistical methods, inference, probability          |
| `mathematics`         | Mathematical ideas, proofs, notation                 |
| `machine-learning`    | ML methods, models, training, evaluation             |
| `reinforcement-learning` | RL-specific topics                               |

A post can carry multiple tags. Prefer fewer — one or two per post is typical.

Category pages and filter UI are suppressed for now. When the archive is large enough to warrant navigation by topic, they can be surfaced by updating `blog/index.qmd` and `styles.css`.
