# Privacy and publication

Scribrism performs inference from local files. The setup command contacts the public model host to fetch a pinned checkpoint; dependency installation contacts package repositories. Those services receive normal network request metadata. No model download occurs during `identity`, `scan` or `verify`.

Generated artifacts are sensitive by default. HTML and extracted-text files contain scanned text. JSON can include input paths, runtime details and error messages. A local hash does not make the underlying document anonymous. Gitignore rules are a precaution, not a security boundary.

For a public release or contribution:

1. Use an explicit list of source, documentation, license and synthetic-test files. Exclude inputs, outputs, research corpora, credentials, environment files, caches, archives and symlinks to private folders.
2. Read every included file and inspect the final Git diff. Search for personal paths, names, contact information, credentials and copied document text. Review tool output and screenshots before attaching them anywhere public.
3. Inspect all history that will become reachable, including author/committer names, email addresses, commit messages, tags, branches and annotated-tag metadata. A clean current tree does not clean old commits.
4. Record the reviewed commit/tree and file hashes privately. After publication, compare the remote tree to that reviewed snapshot. Repeat the gate whenever included files change.

Public repository ownership and the account's public profile remain visible. Sanitizing this project does not conceal an account's existing public identity or activity.

If you discover private information in a proposed contribution, stop publication and report it privately to the repository owner. Do not quote the material in a public issue. Removing it in a later commit does not remove it from history or external copies.
