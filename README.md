# miryusupov.com

Source for my personal site.

I am a machine learning engineer based in Paris, working on retrieval and LLM
systems and the numerical code underneath them. This repository holds the site
itself: short technical notes, and pages for the open-source packages I
maintain.

## Contents

- `index.qmd`, `about.qmd` — site pages
- `blog/` — technical notes
- `projects/` — pages for `robustcov`, `lattice-dsp`, and `cholrot`
- `styles.css`, `_nav.html`, `_footer.html` — layout and styling

The packages themselves live in their own repositories.

## Build

Built with [Quarto](https://quarto.org). Output goes to `docs/` and is served
via GitHub Pages.

```bash
quarto preview   # local, with live reload
quarto render    # write docs/
```

## Independence

This is independent personal work, produced outside employment using public
sources, publicly available software, and personal resources. It does not
represent the views of any employer or client, and contains no employer or
client work product, internal method, data, or confidential know-how.

## Copyright and license

Text, figures, diagrams, notes, explanations, and other non-code content are:

© 2026 Shohruh Miryusupov. All rights reserved, unless otherwise stated.

No license is granted to reproduce, distribute, modify, or create derivative
works from this material except where explicitly stated or where allowed by
applicable law.

If a specific file, note, or code block is released under a separate license,
that license will be stated explicitly.

## Citation

If you refer to a note, please cite the page title, author, URL, and date shown
on that page.

```text
Shohruh Miryusupov, "Title of note," miryusupov.com, YYYY-MM-DD.
Accessed YYYY-MM-DD.
```

## Disclaimer

The notes here are working material. They may contain errors, incomplete
arguments, or ideas still in progress.
